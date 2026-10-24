// One listener answer per request, stored as its own private blob (votes/<day>/<time>-<rand>.json).
// Read them back with ops/fetch_votes.sh. Nothing identifying is stored: a random rater id from the browser only.
import { put } from "@vercel/blob";

const KINDS = new Set(["profile", "rate", "en"]);

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  const b = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
  if (!KINDS.has(b.kind) || typeof b.rater !== "string" || b.rater.length > 40) {
    return res.status(400).json({ error: "bad vote" });
  }
  const row = {
    t: Date.now(),
    kind: b.kind,
    rater: b.rater,
    clip: String(b.clip || "").slice(0, 80),
    natural: clamp(b.natural),
    words: clamp(b.words),
    guess: ["human", "ai"].includes(b.guess) ? b.guess : null,
    native: ["native", "learner", "no"].includes(b.native) ? b.native : null,
    ms: Number.isFinite(b.ms) ? Math.min(b.ms, 600000) : null,
  };
  const day = new Date(row.t).toISOString().slice(0, 10);
  const name = `votes/${day}/${row.t}-${Math.random().toString(36).slice(2, 10)}.json`;
  await put(name, JSON.stringify(row), { access: "private", contentType: "application/json", addRandomSuffix: false });
  res.status(200).json({ ok: true });
}

function clamp(v) {
  const n = Number(v);
  return Number.isInteger(n) && n >= 1 && n <= 5 ? n : null;
}
