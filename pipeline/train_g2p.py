"""Train the learned decision layer of the Nepali G2P on Google's lexicon.

For every word, the rule path is a template with decision points (inherent
schwa keep/drop, gemination, व b/w, nasal place). We brute-force the decision
combinations that reproduce the lexicon pronunciation, then count which choice
each character context prefers. Inference backs off from wide to narrow
context and falls back to the hand rule when no context is confident.

Honest evaluation: words are split 80/10/10 (train/dev/test) by hash.
Hyperparameters are tuned on dev; test is reported once. The shipped model is
then retrained on all words (in-lexicon words are looked up anyway; the model
only matters for unseen words).

Usage:
    python pipeline/train_g2p.py            # tune, report, write data/g2p_model.json
"""
import gzip
import hashlib
import importlib
import itertools
import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
G = importlib.import_module("frontend.g2p")
from pipeline.eval_g2p import edit_distance, norm_google, norm_mine

MAX_COMBOS = 4096


def load_pairs():
    pairs = []
    for line in open(G._LEXICON_PATH, encoding="utf-8"):
        if line.startswith("#") or "\t" not in line:
            continue
        w, p = line.rstrip("\n").split("\t")[:2]
        w = unicodedata.normalize("NFC", w)
        if not w.isascii():
            pairs.append((w, norm_google(p)))
    return pairs


def split_of(word):
    h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % 10
    return "test" if h == 0 else "dev" if h == 1 else "train"


def rule_parts(word):
    return G._rule_parts(word)


def label_word(word, ref):
    parts = rule_parts(word)
    decisions = [(it, chars) for items, chars in parts for it in items if not isinstance(it, str)]
    n_combos = 1
    for it, _ in decisions:
        n_combos *= len(it[2])
    if n_combos > MAX_COMBOS:
        return None
    best = None
    for combo in itertools.product(*[it[2] for it, _ in decisions]):
        out, k = [], 0
        for items, _ in parts:
            m = sum(1 for it in items if not isinstance(it, str))
            out += G._realize(items, list(combo[k:k + m]))
            k += m
        if norm_mine(out) == ref:
            dev = sum(c != it[3] for c, (it, _) in zip(combo, decisions))
            if best is None or dev < best[0]:
                best = (dev, combo)
    if best is None:
        return None
    return [(it[0], chars, it[1], c) for c, (it, chars) in zip(best[1], decisions)]


def count_contexts(labeled):
    counts = defaultdict(Counter)
    for decs in labeled:
        for kind, chars, pos, choice in decs:
            for key in G.context_keys(kind, chars, pos):
                counts[key][choice] += 1
    return counts


def build_model(counts, min_count, min_conf):
    model = {}
    for key, c in counts.items():
        total = sum(c.values())
        if total < min_count:
            continue
        label, n = c.most_common(1)[0]
        if n / total >= min_conf:
            model[key] = label
    return model


def build_vote_model(counts, min_count):
    return {k: dict(c) for k, c in counts.items() if sum(c.values()) >= min_count}


def fold_of(word):
    return int(hashlib.sha1(word.encode("utf-8")).hexdigest(), 16) % 2


def build_selector(pairs, labels, params):
    """Compound-vs-rules stats, cross-fitted: the rules model used to judge a
    word never saw that word, so 'rules' is not unfairly favoured."""
    stats = defaultdict(Counter)
    saved = (G.MODEL, G.MODEL_PARAMS, G.SELECTOR)
    try:
        for k in (0, 1):
            G.MODEL = build_vote_model(count_contexts(
                [labels[w] for w, _ in pairs if labels.get(w) and fold_of(w) != k]), 1)
            G.MODEL_PARAMS, G.SELECTOR = params, {}
            for w, ref in pairs:
                if fold_of(w) != k:
                    continue
                lex = G.LEXICON.pop(w, None)
                try:
                    tag = ""
                    if G._split_suffix(w, G.LEXICON):
                        continue
                    parts = G._compound_split(w)
                    if not parts:
                        continue
                    comp = G.compound_candidate(parts)
                    rules = G._phonemize_word(w, use_lexicon=False)
                    conf = G.rules_confidence(w, comp) if comp != rules else None
                finally:
                    if lex is not None:
                        G.LEXICON[w] = lex
                if comp == rules:
                    continue
                cok, rok = norm_mine(comp) == ref, norm_mine(rules) == ref
                if cok != rok:
                    for key in G.selector_keys(parts, comp, rules, conf, tag):
                        stats[key]["comp" if cok else "rules"] += 1
    finally:
        G.MODEL, G.MODEL_PARAMS, G.SELECTOR = saved
    return {k: dict(v) for k, v in stats.items()}


def score(pairs, model, unseen, params=None, selector=None):
    saved_model, saved_params, saved_sel = G.MODEL, G.MODEL_PARAMS, G.SELECTOR
    G.MODEL, G.MODEL_PARAMS, G.SELECTOR = model, (params or {}), (selector or {})
    ok = perr = ptot = 0
    try:
        for w, ref in pairs:
            if unseen:
                saved = G.LEXICON.pop(w, None)
                try:
                    hyp = G._phonemize_word(w, use_lexicon=True)
                finally:
                    if saved is not None:
                        G.LEXICON[w] = saved
            else:
                hyp = G._phonemize_word(w, use_lexicon=False)
            hyp = norm_mine(hyp)
            ok += hyp == ref
            perr += edit_distance(ref, hyp)
            ptot += len(ref)
    finally:
        G.MODEL, G.MODEL_PARAMS, G.SELECTOR = saved_model, saved_params, saved_sel
    return 100 * ok / len(pairs), 100 * perr / ptot


def main():
    pairs = load_pairs()
    sets = defaultdict(list)
    for w, ref in pairs:
        sets[split_of(w)].append((w, ref))
    print({k: len(v) for k, v in sets.items()})

    labels = {}
    for w, ref in pairs:
        lab = label_word(w, ref)
        if lab is not None:
            labels[w] = lab
    reach = {k: 100 * sum(w in labels for w, _ in v) / len(v) for k, v in sets.items()}
    print(f"words whose lexicon pronunciation is reachable by decisions: {reach}")

    train_counts = count_contexts([labels[w] for w, _ in sets["train"] if w in labels])
    base_dev = score(sets["dev"], {}, unseen=True)
    print(f"dev  baseline (hand rules)   unseen acc {base_dev[0]:.2f}%  PER {base_dev[1]:.2f}%")
    candidates = []
    for min_count in (1, 2, 3):
        for min_conf in (0.6, 0.7):
            candidates.append(({"mode": "first", "min_count": min_count, "min_conf": min_conf},
                               build_model(train_counts, min_count, min_conf)))
    vote_model = build_vote_model(train_counts, 1)
    for alpha in (0.4, 0.5, 0.6, 0.7):
        for smooth in (2, 3, 5):
            candidates.append(({"mode": "vote", "min_count": 1, "alpha": alpha, "smooth": smooth}, vote_model))
    best = None
    for params, model in candidates:
        acc, per = score(sets["dev"], model, unseen=True, params=params)
        print(f"  dev  {params}: acc {acc:.2f}%  PER {per:.2f}%")
        if best is None or (acc, -per) > (best[0], -best[1]):
            best = (acc, per, params, model)
    _, _, params, model = best
    sel_stats = build_selector(sets["train"], labels, params)
    sel_best = None
    for min_n in (3, 5, 10):
        for margin in (1.0, 1.5, 2.0):
            selector = {"stats": sel_stats, "params": {"min_n": min_n, "margin": margin}}
            acc, per = score(sets["dev"], model, True, params, selector)
            print(f"  dev  selector min_n={min_n} margin={margin}: acc {acc:.2f}%  PER {per:.2f}%")
            if sel_best is None or (acc, -per) > (sel_best[0], -sel_best[1]):
                sel_best = (acc, per, selector["params"])
    sel_params = sel_best[2]
    selector = {"stats": sel_stats, "params": sel_params}
    t_base = score(sets["test"], {}, unseen=True)
    t_rules_base = score(sets["test"], {}, unseen=False)
    t_nosel = score(sets["test"], model, unseen=True, params=params)
    t = score(sets["test"], model, unseen=True, params=params, selector=selector)
    t_rules = score(sets["test"], model, unseen=False, params=params)
    print(f"\nTEST (held out, chosen {params}, selector {sel_params})")
    print(f"  without selector: {t_nosel[0]:.2f}% / PER {t_nosel[1]:.2f}%")
    print(f"  unseen pipeline: hand rules {t_base[0]:.2f}% / PER {t_base[1]:.2f}%  ->  learned {t[0]:.2f}% / PER {t[1]:.2f}%")
    print(f"  rules only     : hand rules {t_rules_base[0]:.2f}% / PER {t_rules_base[1]:.2f}%  ->  learned {t_rules[0]:.2f}% / PER {t_rules[1]:.2f}%")

    all_counts = count_contexts(list(labels.values()))
    if params["mode"] == "vote":
        final = build_vote_model(all_counts, params["min_count"])
    else:
        final = build_model(all_counts, params["min_count"], params["min_conf"])
    final_selector = {"stats": build_selector(pairs, labels, params), "params": sel_params}
    meta = {"source": "Google Nepali lexicon (CC-BY 4.0), github.com/google/language-resources",
            "params": {**params, "context_orders": G.CONTEXT_ORDERS},
            "heldout_test": {"unseen_acc": round(t[0], 2), "unseen_per": round(t[1], 2),
                             "hand_rules_unseen_acc": round(t_base[0], 2), "hand_rules_unseen_per": round(t_base[1], 2)},
            "n_contexts": len(final)}
    with gzip.open(G._MODEL_PATH, "wt", encoding="utf-8") as f:
        json.dump({"meta": meta, "decisions": final, "selector": final_selector}, f,
                  ensure_ascii=False, separators=(",", ":"))
    print(f"\nwrote {G._MODEL_PATH} ({len(final)} contexts, {os.path.getsize(G._MODEL_PATH)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
