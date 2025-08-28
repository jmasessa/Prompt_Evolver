"""
GEPA-for-Compliance — Minimal Evolver (v0.3)
Complete version with:
- Evolutionary loop
- Robust LLMClient (OpenAI Chat + Responses API, HF fallback)
- Pareto front summary printing
"""

import os, re, json, time, random, sqlite3, argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# -----------------------------
# Task set (compliance questions)
# -----------------------------
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
# Seed prompt genome
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
# LLM client
# -----------------------------
class LLMClient:
    def __init__(self, model: str):
        self.model = model
        self.oa_key = os.getenv("OPENAI_API_KEY")
        self.org = os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION")
        self.project = os.getenv("OPENAI_PROJECT")
        self.hf_url = os.getenv("HUGGINGFACE_API_URL")
        self.hf_key = os.getenv("HUGGINGFACE_API_KEY")
        if not (self.oa_key or (self.hf_url and self.hf_key)):
            print("[WARN] No LLM credentials found. You can still run dry tests but generation will fail.")

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 400) -> str:
        # --- OpenAI branch ---
        safe_max = max(max_tokens, 16) # GPT-5 requires >=16
        if self.oa_key:
            import requests
            url_chat = "https://api.openai.com/v1/chat/completions"
            url_resp = "https://api.openai.com/v1/responses"
            headers = {"Authorization": f"Bearer {self.oa_key}", "Content-Type": "application/json"}
            if self.org: headers["OpenAI-Organization"] = self.org
            if self.project: headers["OpenAI-Project"] = self.project

            def post_chat():
                if str(self.model).lower().startswith("gpt-5"):
                    payload = {"model": self.model, "input": messages, "max_output_tokens": safe_max}
                else:
                    payload = {"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": safe_max}
                r = requests.post(url_chat, headers=headers, json=payload, timeout=60)
                if r.status_code >= 400:
                    print("[API ERROR/chat]", r.status_code, r.text)
                    r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()

            def post_resp():
                if str(self.model).lower().startswith("gpt-5"):
                    payload = {"model": self.model, "input": messages, "max_output_tokens": safe_max}
                else:
                    payload = {"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": safe_max}
                r = requests.post(url_resp, headers=headers, json=payload, timeout=60)
                if r.status_code >= 400:
                    print("[API ERROR/responses]", r.status_code, r.text)
                    r.raise_for_status()
                data = r.json()
                if isinstance(data, dict) and "output_text" in data:
                    return data["output_text"].strip()
                if "choices" in data:
                    try:
                        return data["choices"][0]["message"]["content"].strip()
                    except Exception:
                        pass
                return (json.dumps(data)[:2000])

            use_responses_first = str(self.model).lower().startswith("gpt-5")
            try:
                return post_resp() if use_responses_first else post_chat()
            except requests.HTTPError as e:
                body = e.response.text.lower() if getattr(e, 'response', None) else ''
                if (not use_responses_first) and ("responses" in body or "use the responses" in body):
                    return post_resp()
                raise RuntimeError(f"OpenAI HTTP {getattr(e.response,'status_code','??')}: {body[:500]}") from e

        # --- HuggingFace branch ---
        if self.hf_url and self.hf_key:
            import requests
            headers = {"Authorization": f"Bearer {self.hf_key}", "Content-Type": "application/json"}
            if str(self.model).lower().startswith("gpt-5"):
                payload = {"inputs": {"messages": messages, "max_new_tokens": max_tokens}}
            else:
                payload = {"inputs": {"messages": messages, "max_new_tokens": max_tokens, "temperature": 0.2}}
            r = requests.post(self.hf_url, headers=headers, json=payload, timeout=60)
            if r.status_code >= 400:
                print("[HF ERROR]", r.status_code, r.text)
                r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            if "choices" in data:
                return data["choices"][0]["message"]["content"].strip()
            return str(data)[:2000]

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
        ordered = ["system", "style", "policy", "answer_rubric"]
        return "\n\n".join([f"[{k.upper()}]\n{self.prompt.get(k,'')}" for k in ordered])

class Archive:
    def __init__(self, path: str = "archive.sqlite"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                parent_a TEXT,
                parent_b TEXT,
                prompt_json TEXT,
                scores_json TEXT
            )
        """)
        self.conn.commit()

    def add(self, cand: Candidate):
        self.conn.execute(
            "INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?,?)",
            (cand.id, time.time(), cand.parent_ids[0], cand.parent_ids[1],
             json.dumps(cand.prompt), json.dumps(cand.scores))
        )
        self.conn.commit()

# -----------------------------
# Mutator (LLM-guided)
# -----------------------------
class Mutator:
    def __init__(self, llm: LLMClient): self.llm = llm

    def mutate(self, cand: Candidate) -> Candidate:
        sys_txt = cand.serialize_prompt()
        instr = ("You are a prompt engineer. Propose a SMALL diff to improve compliance, clarity, and brevity.\n"
                 "Return JSON with keys: 'changes' (list of {section, rewrite}), and 'rationale'.")
        messages = [{"role": "system", "content": instr},
                    {"role": "user", "content": f"CURRENT_PROMPT:\n{sys_txt}"}]
        try:
            resp = self.llm.chat(messages, max_tokens=400)
            data = json.loads(re.search(r"\{[\\s\\S]*\}$", resp).group(0))
            newp = dict(cand.prompt)
            for ch in data.get("changes", []):
                sec = ch.get("section")
                if sec in newp and isinstance(ch.get("rewrite"), str):
                    newp[sec] = ch["rewrite"]
            child_id = f"mut-{int(time.time()*1000)}-{random.randint(1000,9999)}"
            return Candidate(id=child_id, prompt=newp, parent_ids=(cand.id, None))
        except Exception:
            newp = dict(cand.prompt)
            k = random.choice(list(newp.keys()))
            newp[k] += "\nReminder: add a 'Sources' line when citing."
            return Candidate(id=f"mut-fb-{int(time.time()*1000)}", prompt=newp, parent_ids=(cand.id, None))

    def crossover(self, a: Candidate, b: Candidate) -> Candidate:
        newp = {k: random.choice([a.prompt.get(k,''), b.prompt.get(k,'')]) for k in set(a.prompt)|set(b.prompt)}
        return Candidate(id=f"xov-{int(time.time()*1000)}-{random.randint(1000,9999)}",
                         prompt=newp, parent_ids=(a.id,b.id))

# -----------------------------
# Evaluator
# -----------------------------
class Evaluator:
    def __init__(self, llm: LLMClient, model: str):
        self.llm = llm; self.model = model

    def answer(self, cand: Candidate, user_query: str, max_tokens: int) -> str:
        messages = [{"role": "system", "content": cand.serialize_prompt()},
                    {"role": "user", "content": user_query}]
        return self.llm.chat(messages, max_tokens=max_tokens)

    def hard_checks(self, text: str, must: List[str], forbid: List[str], brevity_max_tokens: int) -> Dict[str,float]:
        toks = len(re.findall(r"\\w+|\\S", text))
        brevity = 1.0 if toks <= brevity_max_tokens else max(0.0, 1.0 - (toks-brevity_max_tokens)/(brevity_max_tokens+50))
        has_cite = 1.0 if re.search(r"(?i)sources?:|https?://", text) else 0.0
        forbidden_hit = 1.0 if any(re.search(re.escape(f), text, re.I) for f in forbid) else 0.0
        must_score = sum(1.0 for m in must if re.search(re.escape(m), text, re.I)) / max(1,len(must))
        compliance = 1.0 - forbidden_hit
        return {"brevity": brevity, "has_citation": has_cite, "must_coverage": must_score, "compliance": compliance}

    def llm_judge(self, user: str, answer: str) -> float:
        rubric = ("Rate 0-10. Criteria: clarity, compliance tone, usefulness without personalization, citation where appropriate. Return only a number.")
        msg = [{"role": "system", "content": rubric},
               {"role": "user", "content": f"USER:\n{user}\n\nANSWER:\n{answer}"}]
        out = self.llm.chat(msg, max_tokens=4)
        m = re.search(r"\\d+(?:\\.\\d+)?", out)
        return float(m.group(0)) if m else 5.0

    def evaluate_candidate(self, cand: Candidate) -> Dict[str,float]:
        hard_scores = {"brevity":[], "has_citation":[], "must_coverage":[], "compliance":[]}
        judge_scores = []
        for t in TASKS:
            ans = self.answer(cand, t["user"], t["objectives"]["brevity_max_tokens"])
            hc = self.hard_checks(ans, t["must"], t["forbid"], t["objectives"]["brevity_max_tokens"])
            for k,v in hc.items(): hard_scores[k].append(v)
            judge_scores.append(self.llm_judge(t["user"], ans)/10.0)
        agg = {k: sum(v)/len(v) for k,v in hard_scores.items()}
        agg["judge"] = sum(judge_scores)/len(judge_scores)
        agg["composite"] = 0.35*agg["compliance"] + 0.25*agg["must_coverage"] + 0.2*agg["judge"] + 0.1*agg["has_citation"] + 0.1*agg["brevity"]
        cand.scores = agg
        return agg

# -----------------------------
# Pareto helpers
# -----------------------------
OBJ_KEYS = ["compliance","must_coverage","judge","has_citation","brevity"]
def dominates(a,b): return all(a[k]>=b[k] for k in OBJ_KEYS) and any(a[k]>b[k] for k in OBJ_KEYS)
def pareto_front(cands): return [c for i,c in enumerate(cands) if not any(dominates(o.scores,c.scores) for j,o in enumerate(cands) if j!=i)]

# -----------------------------
# Evolution loop
# -----------------------------
def run(model: str, generations: int, population: int, run_dir: str, verbose: bool = False):
    os.makedirs(run_dir, exist_ok=True)
    llm = LLMClient(model)
    mut = Mutator(llm)
    evl = Evaluator(llm, model)
    arc = Archive()

    pop: List[Candidate] = []
    for i in range(population):
        p = dict(SEED_PROMPT)
        if i>0:
            k = random.choice(list(p.keys()))
            p[k] += "\nWrite in bullet points when listing steps."
        c = Candidate(id=f"seed-{i}", prompt=p)
        c.scores = evl.evaluate_candidate(c)
        arc.add(c); pop.append(c)

    for g in range(generations):
        print(f"\n=== Generation {g+1}/{generations} ===")
        front = pareto_front(pop)
        print(f"Current Pareto front size: {len(front)}")
        if verbose:
            for c in front:
                s=c.scores
                print(f"  {c.id} | comp={s['composite']:.2f} must={s['must_coverage']:.2f} judge={s['judge']:.2f} cite={s['has_citation']:.2f} brev={s['brevity']:.2f}")
        children=[]
        while len(children)<population:
            if random.random()<0.6 and front:
                parent=random.choice(front)
                child=mut.mutate(parent)
            else:
                a,b=random.sample(pop,2)
                child=mut.crossover(a,b)
            child.scores=evl.evaluate_candidate(child)
            arc.add(child); children.append(child)
        merged=pop+children
        merged.sort(key=lambda c:c.scores["composite"],reverse=True)
        top=merged[:max(2,population//2)]
        pf=pareto_front(merged)
        diversity=[c for c in pf if c not in top]
        random.shuffle(diversity)
        pop=(top+diversity)[:population]

    print("Done. See run directory and archive.sqlite.")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--run_dir", type=str, default=f"runs/{int(time.time())}")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args.model, args.generations, args.population, args.run_dir, args.verbose)
