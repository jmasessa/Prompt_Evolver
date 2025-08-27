"""
GEPA-for-Compliance — Minimal Evolver (v0.1)

A tiny evolutionary framework to evolve prompts for a compliance-savvy assistant.
- True Evolutionary AI: mutation & crossover are *LLM-guided* (AI proposes prompt diffs),
  evaluation includes *LLM-as-judge* alongside hard checks.
- Easy to vibe-code: single file; swap in your preferred LLM provider (OpenAI, local HF, vLLM, etc.).
- Extensible to #4 (UI-TARS macro evolver) with the same loop (see TODO at bottom).

Run (example):
  $ python evolver.py --model gpt-4o-mini --generations 5 --population 8

Env vars:
  OPENAI_API_KEY=<key>            # if using OpenAI
  HUGGINGFACE_API_URL=<url>       # optional alt; e.g., your ui-tars endpoint
  HUGGINGFACE_API_KEY=<key>

Outputs:
  ./archive.sqlite  — candidate lineage, scores, and prompts
  ./runs/<timestamp>/ — JSONL of candidates + leaderboard
"""

import os
import re
import json
import time
import math
import random
import sqlite3
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple

# -----------------------------
# Small task set (seed corpus)
# -----------------------------
# Each item provides: a user query, constraints (must/forbid), and meta objectives.
# NOTE: These are illustrative, not investment advice.
TASKS = [
    {
        "id": "fees_disclosure",
        "user": "What are the expense ratios for Admiral Shares of Vanguard Total Stock Market Index?",
        "must": ["cite source", "disclose that data can change"],
        "forbid": ["personalized advice", "performance guarantees"],
        "objectives": {"brevity_max_tokens": 140, "clarity": 1.0}
    },
    {
        "id": "tax_cost_basis",
        "user": "How do I find my cost basis for tax purposes?",
        "must": ["education only disclaimer", "steps a client can do on site"],
        "forbid": ["tax advice", "legal advice"],
        "objectives": {"brevity_max_tokens": 180, "clarity": 1.0}
    },
    {
        "id": "retirement_withdrawal",
        "user": "Should I move my bond fund into a CD for retirement next year?",
        "must": ["explain cannot provide personalized advice", "offer general factors", "suggest contacting advisor"],
        "forbid": ["specific allocation", "timing calls"],
        "objectives": {"brevity_max_tokens": 160, "clarity": 1.0}
    },
]

# -----------------------------
# Seed prompt genome (sections)
# -----------------------------
SEED_PROMPT = {
    "system": (
        "You are a compliance-aware financial information assistant. "
        "Always avoid personalized investment, tax, or legal advice. "
        "Prefer concise, plain-English answers with bullet lists when helpful. "
        "Cite a credible source link when referencing data (fees, policies)."
    ),
    "style": (
        "Tone: neutral, educational. Include an 'Education-only' disclaimer if the user asks for advice. "
        "If uncertain, ask the user to consult Vanguard resources or a qualified professional."
    ),
    "policy": (
        "NEVER provide specific allocations, guarantees, or personalized recommendations. "
        "Flag ambiguity and ask clarifying questions before proceeding."
    ),
    "answer_rubric": (
        "1) Check for prohibited content. 2) Add disclaimers when appropriate. "
        "3) Provide 1–2 actionable, generic steps. 4) Add a 'Sources' line when citing data."
    ),
}

# -----------------------------
# LLM client (pluggable)
# -----------------------------
class LLMClient:
    def __init__(self, model: str):
        self.model = model
        self.oa_key = os.getenv("OPENAI_API_KEY")
        #OPENAI_API_KEY = "PUT OPEN AI KEY HERE"
        #self.oa_key = OPENAI_API_KEY
        self.hf_url = os.getenv("HUGGINGFACE_API_URL")
        self.hf_key = os.getenv("HUGGINGFACE_API_KEY")
        if not (self.oa_key or (self.hf_url and self.hf_key)):
            print("[WARN] No LLM credentials found. You can still run dry tests but generation will fail.")

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 400) -> str:
        """Minimal chat wrapper. Uses OpenAI if key present; else HuggingFace Inference Endpoint; else raises."""
        if self.oa_key:
            import requests
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.oa_key}", "Content-Type": "application/json"}
            payload = {"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens}
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        elif self.hf_url and self.hf_key:
            import requests
            headers = {"Authorization": f"Bearer {self.hf_key}", "Content-Type": "application/json"}
            payload = {"inputs": {"messages": messages, "max_new_tokens": max_tokens, "temperature": 0.2}}
            r = requests.post(self.hf_url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            # Expect either text or OpenAI-style; adjust as needed for your endpoint
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()
            elif isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            else:
                # Fallback: try OpenAI-compatible schema
                return data["choices"][0]["message"]["content"].strip()
        else:
            raise RuntimeError("No LLM provider configured.")

# -----------------------------
# Candidate genome & archive
# -----------------------------
@dataclass
class Candidate:
    id: str
    prompt: Dict[str, str]
    parent_ids: Tuple[str, str] = field(default_factory=lambda: (None, None))
    scores: Dict[str, float] = field(default_factory=dict)

    def serialize_prompt(self) -> str:
        # Concatenate sections into a single system message
        ordered = ["system", "style", "policy", "answer_rubric"]
        return "\n\n".join([f"[{k.upper()}]\n{self.prompt.get(k, '')}" for k in ordered])

class Archive:
    def __init__(self, path: str = "archive.sqlite"):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                parent_a TEXT,
                parent_b TEXT,
                prompt_json TEXT,
                scores_json TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, cand: Candidate):
        self.conn.execute(
            "INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?,?)",
            (
                cand.id,
                time.time(),
                cand.parent_ids[0],
                cand.parent_ids[1],
                json.dumps(cand.prompt),
                json.dumps(cand.scores),
            ),
        )
        self.conn.commit()

    def load_all(self) -> List[Candidate]:
        rows = self.conn.execute("SELECT id,parent_a,parent_b,prompt_json,scores_json FROM candidates").fetchall()
        out = []
        for r in rows:
            out.append(Candidate(
                id=r[0], parent_ids=(r[1], r[2]), prompt=json.loads(r[3]), scores=json.loads(r[4] or '{}')
            ))
        return out

# -----------------------------
# Mutation & crossover (LLM-guided)
# -----------------------------
class Mutator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def mutate(self, cand: Candidate) -> Candidate:
        sys_txt = cand.serialize_prompt()
        instr = (
            "You are a prompt engineer. Propose a SMALL diff to improve compliance, clarity, and brevity.\n"
            "Return JSON with keys: 'changes' (list of {section, rewrite}), and 'rationale'. Keep sections minimal."
        )
        messages = [
            {"role": "system", "content": instr},
            {"role": "user", "content": f"CURRENT_PROMPT:\n{sys_txt}"},
        ]
        try:
            resp = self.llm.chat(messages, max_tokens=400)
            data = json.loads(self._extract_json(resp))
            newp = dict(cand.prompt)
            for ch in data.get("changes", []):
                sec = ch.get("section")
                if sec in newp and isinstance(ch.get("rewrite"), str):
                    newp[sec] = ch["rewrite"]
            child_id = f"mut-{int(time.time()*1000)}-{random.randint(1000,9999)}"
            return Candidate(id=child_id, prompt=newp, parent_ids=(cand.id, None))
        except Exception as e:
            # Fallback: light heuristic mutation if JSON parse/LLM fails
            newp = dict(cand.prompt)
            k = random.choice(list(newp.keys()))
            newp[k] = newp[k] + "\nReminder: add a 'Sources' line when citing."
            child_id = f"mut-fb-{int(time.time()*1000)}-{random.randint(1000,9999)}"
            return Candidate(id=child_id, prompt=newp, parent_ids=(cand.id, None))

    def crossover(self, a: Candidate, b: Candidate) -> Candidate:
        newp = {}
        for k in set(list(a.prompt.keys()) + list(b.prompt.keys())):
            pick = random.choice([a.prompt.get(k, ''), b.prompt.get(k, '')])
            newp[k] = pick
        child_id = f"xov-{int(time.time()*1000)}-{random.randint(1000,9999)}"
        return Candidate(id=child_id, prompt=newp, parent_ids=(a.id, b.id))

    def _extract_json(self, txt: str) -> str:
        # naive JSON block extractor
        m = re.search(r"\{[\s\S]*\}$", txt.strip())
        return m.group(0) if m else txt

# -----------------------------
# Evaluator: hard checks + LLM-as-judge
# -----------------------------
class Evaluator:
    def __init__(self, llm: LLMClient, model: str):
        self.llm = llm
        self.model = model

    def answer(self, cand: Candidate, user_query: str, max_tokens: int) -> str:
        messages = [
            {"role": "system", "content": cand.serialize_prompt()},
            {"role": "user", "content": user_query},
        ]
        return self.llm.chat(messages, max_tokens=max_tokens)

    def hard_checks(self, text: str, must: List[str], forbid: List[str], brevity_max_tokens: int) -> Dict[str, float]:
        toks = len(re.findall(r"\w+|\S", text))
        brevity = 1.0 if toks <= brevity_max_tokens else max(0.0, 1.0 - (toks - brevity_max_tokens) / (brevity_max_tokens+50))
        has_cite = 1.0 if re.search(r"(?i)sources?:|https?://", text) else 0.0
        # Simple keyword guards; adjust to your policy deck
        forbidden_hit = 1.0 if any(re.search(self._to_regex(f), text) for f in forbid) else 0.0
        must_score = sum(1.0 for m in must if re.search(self._to_regex(m), text)) / max(1, len(must))
        compliance = 1.0 - forbidden_hit  # 1 if no forbidden terms matched
        return {"brevity": brevity, "has_citation": has_cite, "must_coverage": must_score, "compliance": compliance}

    def llm_judge(self, user: str, answer: str) -> float:
        rubric = (
            "Rate 0-10. Criteria: (1) clarity, (2) compliance tone, (3) usefulness without personalization, "
            "(4) citation where appropriate. Return only a number."
        )
        msg = [
            {"role": "system", "content": rubric},
            {"role": "user", "content": f"USER:\n{user}\n\nANSWER:\n{answer}"},
        ]
        out = self.llm.chat(msg, max_tokens=4)
        m = re.search(r"\d+(?:\.\d+)?", out)
        return float(m.group(0)) if m else 5.0

    def evaluate_candidate(self, cand: Candidate) -> Dict[str, float]:
        # Multi-objective score averaged across tasks; you could keep per-task too
        hard_scores = {"brevity": [], "has_citation": [], "must_coverage": [], "compliance": []}
        judge_scores = []
        for t in TASKS:
            ans = self.answer(cand, t["user"], t["objectives"]["brevity_max_tokens"]) 
            hc = self.hard_checks(ans, t["must"], t["forbid"], t["objectives"]["brevity_max_tokens"]) 
            for k,v in hc.items(): hard_scores[k].append(v)
            judge_scores.append(self.llm_judge(t["user"], ans) / 10.0)
        # Aggregate (mean)
        agg = {k: sum(v)/len(v) for k,v in hard_scores.items()}
        agg["judge"] = sum(judge_scores)/len(judge_scores)
        # Create composite for leaderboard (still store individual metrics)
        # Weighted: compliance 0.35, must_coverage 0.25, judge 0.2, has_citation 0.1, brevity 0.1
        agg["composite"] = (
            0.35*agg["compliance"] + 0.25*agg["must_coverage"] + 0.2*agg["judge"] + 0.1*agg["has_citation"] + 0.1*agg["brevity"]
        )
        return agg

    def _to_regex(self, phrase: str) -> str:
        # loose match for simple demo
        return re.escape(phrase).replace("\\ ", r"\s+")

# -----------------------------
# Pareto selection helpers
# -----------------------------
OBJ_KEYS = ["compliance", "must_coverage", "judge", "has_citation", "brevity"]

def dominates(a: Dict[str,float], b: Dict[str,float]) -> bool:
    ge_all = all(a[k] >= b[k] for k in OBJ_KEYS)
    gt_any = any(a[k] > b[k] for k in OBJ_KEYS)
    return ge_all and gt_any

def pareto_front(cands: List[Candidate]) -> List[Candidate]:
    front = []
    for i,c in enumerate(cands):
        if not any(dominates(other.scores, c.scores) for j,other in enumerate(cands) if j!=i):
            front.append(c)
    return front

# -----------------------------
# Evolution loop
# -----------------------------

def run(model: str, generations: int, population: int, run_dir: str):
    os.makedirs(run_dir, exist_ok=True)
    llm = LLMClient(model)
    mut = Mutator(llm)
    evl = Evaluator(llm, model)
    arc = Archive()

    # init population from seed with light jitter
    pop: List[Candidate] = []
    for i in range(population):
        p = dict(SEED_PROMPT)
        if i>0:
            # tiny random tweak to seed diversity
            k = random.choice(list(p.keys()))
            p[k] = p[k] + "\nWrite in bullet points when listing steps."
        c = Candidate(id=f"seed-{i}", prompt=p)
        c.scores = evl.evaluate_candidate(c)
        arc.add(c)
        pop.append(c)
    save_run(pop, run_dir)

    for g in range(generations):
        print(f"\n=== Generation {g+1}/{generations} ===")
        front = pareto_front(pop)
        print(f"Current Pareto front size: {len(front)}")
        children: List[Candidate] = []
        # generate offspring
        while len(children) < population:
            if random.random() < 0.6 and len(pop) >= 1:
                parent = random.choice(front if front else pop)
                child = mut.mutate(parent)
            else:
                a, b = random.sample(pop, 2)
                child = mut.crossover(a, b)
            child.scores = evl.evaluate_candidate(child)
            arc.add(child)
            children.append(child)
        # survivor selection: keep best by composite, but ensure Pareto diversity
        merged = pop + children
        merged.sort(key=lambda c: c.scores["composite"], reverse=True)
        # keep top-K by composite, with a few random Pareto picks to maintain diversity
        top = merged[: max(2, population // 2)]
        pf = pareto_front(merged)
        diversity = [c for c in pf if c not in top]
        random.shuffle(diversity)
        pop = (top + diversity)[:population]
        save_run(pop, run_dir)

    print("Done. See run directory and archive.sqlite.")

# -----------------------------
# IO helpers
# -----------------------------

def save_run(pop: List[Candidate], run_dir: str):
    stamp = int(time.time())
    recs = [
        {
            "id": c.id,
            "parents": list(c.parent_ids),
            "scores": c.scores,
            "prompt": c.prompt,
        }
        for c in pop
    ]
    with open(os.path.join(run_dir, f"gen-{stamp}.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    # write current leaderboard
    lb = sorted(pop, key=lambda c: c.scores.get("composite",0.0), reverse=True)
    with open(os.path.join(run_dir, "leaderboard.json"), "w", encoding="utf-8") as f:
        json.dump([
            {"id": c.id, "composite": round(c.scores.get("composite",0.0),4), **{k: round(c.scores.get(k,0.0),4) for k in OBJ_KEYS}}
            for c in lb
        ], f, indent=2)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--run_dir", type=str, default=f"runs/{int(time.time())}")
    args = parser.parse_args()
    run(args.model, args.generations, args.population, args.run_dir)

"""
NEXT: Mapping this to #4 (UI‑TARS macro evolver)
— Replace TASKS with UI flows (e.g., “download statement”, “find expense ratio”).
— Evaluator.success = actual automation run success + duration + retries.
— Mutator operates over macro scripts (JSON actions). LLM mutation prompt -> propose small diffs
  (e.g., add selector fallback, waitFor element, alternate navigation path). Crossover splices
  subsequences that worked.
— Keep per‑page‑state archive: choose parents from same detected DOM signature for robustness.
"""
