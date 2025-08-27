import sqlite3, json, textwrap

"""
conn = sqlite3.connect("archive.sqlite")
rows = conn.execute("SELECT id, parent_a, parent_b, scores_json FROM candidates ORDER BY timestamp DESC LIMIT 5").fetchall()
for r in rows:
    print(r[0], r[1], r[2], json.loads(r[3]))
"""

conn = sqlite3.connect("archive.sqlite")
rows = conn.execute("""
  SELECT id, parent_a, parent_b, prompt_json, scores_json
  FROM candidates
  ORDER BY timestamp DESC
  LIMIT 50
""").fetchall()

# Sort those by composite score (desc) and keep top 5
def composite(scores): return json.loads(scores).get("composite", 0.0)
rows.sort(key=lambda r: composite(r[4]), reverse=True)
top = rows[:5]

for i, (cid, pa, pb, pjson, sjson) in enumerate(top, 1):
    prompt = json.loads(pjson)
    scores = json.loads(sjson)
    print(f"\n#{i}  {cid}  composite={scores['composite']:.3f}")
    print(f"   parents: {pa} | {pb}")
    for sec in ["system","style","policy","answer_rubric"]:
        if sec in prompt:
            print(f"\n[{sec.upper()}]")
            print(textwrap.fill(prompt[sec], width=100))
