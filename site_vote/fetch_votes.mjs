// Download every vote from the private Blob store into votes.jsonl and print a summary per clip kind.
// Usage (from site_vote/, after `vercel env pull`): node --env-file=.env.local fetch_votes.mjs
import { get, list } from "@vercel/blob";
import { readFileSync, writeFileSync } from "node:fs";

const rows = [];
let cursor;
do {
  const page = await list({ prefix: "votes/", cursor, limit: 1000 });
  for (const b of page.blobs) {
    const r = await get(b.pathname, { access: "private", useCache: false });
    if (r && r.statusCode === 200) rows.push(JSON.parse(await new Response(r.stream).text()));
  }
  cursor = page.hasMore ? page.cursor : undefined;
} while (cursor);
writeFileSync("votes.jsonl", rows.map(r => JSON.stringify(r)).join("\n") + "\n");

const M = JSON.parse(readFileSync("clips/manifest.json", "utf8"));
const kindOf = {};
for (const s of M.sentences) { kindOf[s.human.id] = "human"; kindOf[s.swar.id] = s.swar.voice === "female" ? "swar_maya" : "swar_aarav"; }
for (const e of M.english) kindOf[e.id] = e.voice === "female" ? "en_maya" : "en_aarav";
const g = {};
for (const r of rows.filter(r => r.kind !== "profile")) {
  const k = kindOf[r.clip] || "unknown";
  (g[k] ||= { n: 0, nat: 0, words: 0, saidHuman: 0 });
  g[k].n++; g[k].nat += r.natural || 0; g[k].words += r.words || 0; g[k].saidHuman += r.guess === "human" ? 1 : 0;
}
const raters = new Set(rows.map(r => r.rater)).size;
console.log(`${rows.length} rows from ${raters} listeners`);
for (const [k, v] of Object.entries(g)) {
  console.log(`${k.padEnd(11)} n=${String(v.n).padStart(4)}  natural ${(v.nat / v.n).toFixed(2)}  words ${(v.words / v.n).toFixed(2)}` +
    (k.startsWith("en") ? "" : `  heard as human ${(100 * v.saidHuman / v.n).toFixed(0)}%`));
}
