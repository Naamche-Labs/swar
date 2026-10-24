#!/usr/bin/env python3
"""Build an arXiv-ready (pdfLaTeX-safe) copy of the Swar paper.

arXiv cannot run XeTeX/fontspec, so every \\np{...} Devanagari word is rendered
once to a tight PDF (same font, 10pt, fixed baseline strut) and replaced by
\\includegraphics. The preamble switches fontspec/TeX Gyre Termes for the
pdfLaTeX `times` setup used by the tokenizer paper's arXiv build.

Output: arxiv_submit/ (sources), arxiv_submit.tar.gz (upload), arxiv_preview.pdf.
Usage: python3 make_arxiv.py   (needs tectonic)
"""
import glob
import hashlib
import os
import re
import shutil
import subprocess
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "arxiv_submit")
DEV = os.path.join(OUT, "dev")
# strut depth/height keep every image on the same baseline as the text
DEPTH, HEIGHT = "0.45em", "1.35em"
NP = re.compile(r"\\np\{([^{}]*)\}")

STANDALONE = r"""\documentclass[10pt,border=0pt]{standalone}
\usepackage{fontspec}
\newfontfamily\devanagarifont[Script=Devanagari]{Kohinoor Devanagari}
\begin{document}
{\devanagarifont\rule[-%s]{0pt}{%s}%s}
\end{document}
"""

PREAMBLE_OLD = [
    r"\usepackage{fontspec}",
    r"\setmainfont{texgyretermes}",
    r"\newfontfamily\devanagarifont",
    r"\newcommand{\np}",
]
PREAMBLE_NEW = "\n".join([
    r"\usepackage[T1]{fontenc}",
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage{times}",
    r"\newcommand{\devimg}[1]{\raisebox{-%s}{\includegraphics{dev/#1}}}" % DEPTH,
])


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def render(word):
    name = "d" + hashlib.sha1(word.encode()).hexdigest()[:10]
    pdf = os.path.join(DEV, name + ".pdf")
    if not os.path.exists(pdf):
        tmp = os.path.join(DEV, name + ".tex")
        open(tmp, "w", encoding="utf-8").write(STANDALONE % (DEPTH, HEIGHT, word))
        run(["tectonic", "-X", "compile", name + ".tex"], DEV)
        os.remove(tmp)
    return name


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(DEV)
    shutil.copytree(os.path.join(HERE, "sections"), os.path.join(OUT, "sections"))
    for f in ["refs.bib", "IEEEtran.cls"] + glob.glob(os.path.join(HERE, "fig_*.pdf")):
        shutil.copy(os.path.join(HERE, os.path.basename(f)), OUT)

    words = {}
    tex_files = [os.path.join(OUT, "main.tex")] + sorted(glob.glob(os.path.join(OUT, "sections", "*.tex")))
    shutil.copy(os.path.join(HERE, "main.tex"), tex_files[0])
    for f in tex_files:
        t = open(f, encoding="utf-8").read()
        t = NP.sub(lambda m: r"\devimg{%s}" % words.setdefault(m.group(1), render(m.group(1))), t)
        if f.endswith("main.tex"):
            lines = [l for l in t.split("\n") if not any(l.startswith(p) for p in PREAMBLE_OLD)]
            i = next(k for k, l in enumerate(lines) if l.startswith(r"\IEEEoverridecommandlockouts")) + 1
            lines.insert(i, PREAMBLE_NEW)
            t = "\n".join(lines)
        open(f, "w", encoding="utf-8").write(t)

    left = [f for f in tex_files if re.search(r"[\u0900-\u097F]", open(f, encoding="utf-8").read())]
    if left:
        raise SystemExit(f"Devanagari left outside \\np{{}} in: {left}")

    run(["tectonic", "-X", "compile", "--keep-intermediates", "main.tex"], OUT)
    shutil.copy(os.path.join(OUT, "main.pdf"), os.path.join(HERE, "arxiv_preview.pdf"))

    keep = (".tex", ".bbl", ".cls", ".pdf")
    with tarfile.open(os.path.join(HERE, "arxiv_submit.tar.gz"), "w:gz") as tar:
        for root, _, files in os.walk(OUT):
            for n in files:
                if n.endswith(keep) and n != "main.pdf":
                    p = os.path.join(root, n)
                    tar.add(p, arcname=os.path.relpath(p, OUT))
    print(f"{len(words)} Devanagari words rendered; arxiv_preview.pdf and arxiv_submit.tar.gz written")


if __name__ == "__main__":
    main()
