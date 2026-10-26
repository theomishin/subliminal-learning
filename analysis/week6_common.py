"""Shared pieces for the week-6 experiments (Sam's list from the Wed 21 Oct meeting).

Nothing here imports Unsloth or vLLM, so every function runs on a CPU with plain
transformers + peft. The training script imports Unsloth itself, first, when it
uses that backend.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WEEK6 = DATA / "week6"
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"
TOKENIZER_DIR = DATA / "models/qwen_2.5_7b-owl_numbers-s1"
ALL_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")
ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
HIDDEN_TARGETS = ("gate_proj", "up_proj")  # read the residual stream, make the MLP's hidden activation
READOUT_TARGETS = ("down_proj",)  # write the hidden activation back into the residual stream
N_LAYERS = 28
# First tokens of the common owl answers (W12 in CODEX_KERNEL.md) plus the full answers we
# teacher-force. "Owl" is two tokens in Qwen's vocabulary ("O" + "wl"); "owl" is one.
OWL_FIRST_TOKENS = ("O", "owl", "OWL")
OWL_ANSWERS = ("Owl", "owl")


def save_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def parse_layers(text: str | None) -> list[int] | None:
    """'0-6' or '0,1,2' or '0-2,5' -> sorted list; None/'' /'all' -> None (every layer)."""
    if text is None or text.strip() in ("", "all"):
        return None
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def parse_modules(text: str | None) -> list[str]:
    if text is None or text.strip() in ("", "all"):
        return list(ALL_TARGETS)
    named = {"mlp": MLP_TARGETS, "attn": ATTN_TARGETS, "hidden": HIDDEN_TARGETS,
             "readout": READOUT_TARGETS}
    if text.strip() in named:
        return list(named[text.strip()])
    mods = [m.strip() for m in text.split(",") if m.strip()]
    bad = [m for m in mods if m not in ALL_TARGETS]
    if bad:
        raise ValueError(f"Unknown modules {bad}; allowed {ALL_TARGETS} or mlp/attn/hidden/readout")
    return mods


def question_ci(per_question_rates: list[float]) -> dict:
    """The paper's interval: normal CI over question-level rates."""
    mean = statistics.mean(per_question_rates)
    if len(per_question_rates) > 1:
        margin = statistics.NormalDist().inv_cdf(0.975) * statistics.stdev(per_question_rates) / math.sqrt(len(per_question_rates))
    else:
        margin = float("nan")
    return {"mean": mean, "lower_bound": mean - margin, "upper_bound": mean + margin}


def owl_rate_from_jsonl(path: Path) -> dict:
    """Read a run_evaluation.py output (one line per question) and count 'owl' answers."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rates, counts, total = [], 0, 0
    for row in rows:
        answers = [r["response"]["completion"] for r in row["responses"]]
        hits = sum("owl" in a.lower() for a in answers)
        rates.append(hits / len(answers))
        counts += hits
        total += len(answers)
    return {"questions": len(rows), "answers": total, "owl_answers": counts,
            "owl_rate": counts / total if total else float("nan"), "ci": question_ci(rates)}


def load_tokenizer(path: Path | str = TOKENIZER_DIR):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if tok.pad_token_id is None:
        raise ValueError("tokenizer must define a pad token")
    return tok


def animal_questions() -> list[str]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("traits", ROOT / "cfgs/preference_numbers/traits.py")
    traits = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(traits)
    return list(traits.ANIMAL_QUESTIONS)


def training_rows(path: Path, count: int = 10_000, seed: int = 1) -> list[dict]:
    """Same sampling as sl/finetuning/services.py::run_finetuning_job."""
    import random
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) > count:
        rows = random.Random(seed).sample(rows, count)
    return rows


class OwlProbe:
    """Owl logit / probability at the 50 animal questions, read in one batched forward.

    Objects: q = one animal question formatted as a chat with the generation prompt;
    the model's logits at the last position of q are the scores for the first answer
    token. We record, per question: the logit and probability of each owl first-token
    (O, owl, OWL), their summed probability, the log-probability of the whole answers
    "Owl<|im_end|>" and "owl<|im_end|>" by teacher forcing, and the rank of "O" among
    all tokens.
    """

    def __init__(self, tokenizer, questions: list[str], device=None, batch_size: int = 25):
        self.tok = tokenizer
        self.questions = questions
        self.batch_size = batch_size
        self.device = device
        self.first_ids = {}
        for word in OWL_FIRST_TOKENS:
            ids = tokenizer.encode(word, add_special_tokens=False)
            if len(ids) == 1:
                self.first_ids[word] = ids[0]
        self.im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}],
                                                      tokenize=True, add_generation_prompt=True)
                        for q in questions]
        self.answers = {a: tokenizer.encode(a, add_special_tokens=False) + [self.im_end] for a in OWL_ANSWERS}

    def _pad(self, seqs: list[list[int]]):
        length = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), length), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), length), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        return ids, mask

    @torch.no_grad()
    def measure(self, model) -> dict:
        was_training = model.training
        model.eval()
        device = self.device or next(model.parameters()).device
        n = len(self.prompts)
        out = {"first_logit": {w: [0.0] * n for w in self.first_ids},
               "first_prob": {w: [0.0] * n for w in self.first_ids},
               "owl_start_prob": [0.0] * n,
               "answer_logprob": {a: [0.0] * n for a in self.answers},
               "rank_O": [0] * n, "top1_token": [""] * n, "top1_prob": [0.0] * n}
        for start in range(0, n, self.batch_size):
            idx = list(range(start, min(start + self.batch_size, n)))
            # first-token scores: right padding, read the last real position
            ids, mask = self._pad([self.prompts[i] for i in idx])
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False).logits
            last = (mask.sum(1) - 1).to(device)
            scores = logits[torch.arange(len(idx), device=device), last].float()
            probs = F.softmax(scores, dim=-1)
            for w, tid in self.first_ids.items():
                for k, i in enumerate(idx):
                    out["first_logit"][w][i] = scores[k, tid].item()
                    out["first_prob"][w][i] = probs[k, tid].item()
            mass = sum(probs[:, tid] for tid in self.first_ids.values())
            for k, i in enumerate(idx):
                out["owl_start_prob"][i] = mass[k].item()
                if "O" in self.first_ids:
                    out["rank_O"][i] = int((scores[k] > scores[k, self.first_ids["O"]]).sum().item()) + 1
                top = probs[k].argmax().item()
                out["top1_token"][i] = self.tok.decode([top])
                out["top1_prob"][i] = probs[k, top].item()
            # whole answers by teacher forcing
            for answer, a_ids in self.answers.items():
                seqs = [self.prompts[i] + a_ids for i in idx]
                ids2, mask2 = self._pad(seqs)
                lg = model(input_ids=ids2.to(device), attention_mask=mask2.to(device), use_cache=False).logits.float()
                logp = F.log_softmax(lg, dim=-1)
                for k, i in enumerate(idx):
                    p0 = len(self.prompts[i])
                    total = 0.0
                    for j, t in enumerate(a_ids):
                        total += logp[k, p0 - 1 + j, t].item()
                    out["answer_logprob"][answer][i] = total
        if was_training:
            model.train()
        summary = {
            "mean_first_logit": {w: statistics.mean(v) for w, v in out["first_logit"].items()},
            "mean_first_prob": {w: statistics.mean(v) for w, v in out["first_prob"].items()},
            "mean_owl_start_prob": statistics.mean(out["owl_start_prob"]),
            "mean_answer_logprob": {a: statistics.mean(v) for a, v in out["answer_logprob"].items()},
            "mean_answer_prob": {a: statistics.mean(math.exp(x) for x in v) for a, v in out["answer_logprob"].items()},
            "median_rank_O": statistics.median(out["rank_O"]),
        }
        return {"per_question": out, "summary": summary}


def rises_between_steps(track: list[dict], key_path: tuple[str, ...]) -> list[float]:
    """Fraction of questions whose tracked value rose from step t-1 to step t, per step t>=1."""
    def get(entry):
        v = entry["per_question"]
        for k in key_path:
            v = v[k]
        return v
    fractions = []
    for prev, cur in zip(track[:-1], track[1:]):
        a, b = get(prev), get(cur)
        fractions.append(sum(y > x for x, y in zip(a, b)) / len(a))
    return fractions


# ---------------------------------------------------------------- HF sampling (no vLLM)

@torch.no_grad()
def hf_sample_answers(model, tokenizer, questions: list[str], n_samples: int, *, batch_size: int = 200,
                      max_new_tokens: int = 64, temperature: float = 1.0, seed: int = 1,
                      step_hook=None, device=None) -> list[dict]:
    """Sample answers with the ordinary HF model and a KV cache, left-padded prompts.

    step_hook(input_ids, attention_mask, position_ids, past) -> logits lets the decoupled
    forward (week6_decouple.py) replace the model call; None means a plain forward.
    Returns run_evaluation.py's shape: one dict per question with "responses".
    """
    device = device or next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    eos_ids = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id}
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=True,
                                             add_generation_prompt=True) for q in questions]
    jobs = [(qi, s) for qi in range(len(questions)) for s in range(n_samples)]
    answers: dict[int, list[str]] = {qi: [] for qi in range(len(questions))}
    model.eval()
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        seqs = [prompts[qi] for qi, _ in chunk]
        length = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), length), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), length), dtype=torch.long)
        for i, s in enumerate(seqs):  # left padding
            ids[i, length - len(s):] = torch.tensor(s)
            mask[i, length - len(s):] = 1
        ids, mask = ids.to(device), mask.to(device)
        past = None
        generated = torch.zeros((len(seqs), 0), dtype=torch.long, device=device)
        finished = torch.zeros(len(seqs), dtype=torch.bool, device=device)
        cur = ids
        for _ in range(max_new_tokens):
            pos = (mask.cumsum(1) - 1).clamp(min=0)[:, -cur.shape[1]:]
            if step_hook is None:
                out = model(input_ids=cur, attention_mask=mask, position_ids=pos, past_key_values=past, use_cache=True)
                logits, past = out.logits[:, -1].float(), out.past_key_values
            else:
                logits, past = step_hook(cur, mask, pos, past)
                logits = logits.float()
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            nxt = torch.where(finished, torch.full_like(nxt, tokenizer.pad_token_id), nxt)
            generated = torch.cat([generated, nxt[:, None]], dim=1)
            for e in eos_ids:
                finished |= nxt == e
            if finished.all():
                break
            cur = nxt[:, None]
            mask = torch.cat([mask, torch.ones((len(seqs), 1), dtype=torch.long, device=device)], dim=1)
        for i, (qi, _) in enumerate(chunk):
            toks = generated[i].tolist()
            cut = []
            for t in toks:
                if t in eos_ids or t == tokenizer.pad_token_id:
                    break
                cut.append(t)
            answers[qi].append(tokenizer.decode(cut))
        logger.info(f"sampled {min(start + batch_size, len(jobs))}/{len(jobs)}")
    return [{"question": q, "responses": [{"response": {"completion": a}} for a in answers[qi]]}
            for qi, q in enumerate(questions)]


def write_eval_jsonl(path: Path, groups: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for g in groups:
            f.write(json.dumps(g) + "\n")


# ---------------------------------------------------------------- tiny model for CPU gates

def tiny_model(tokenizer, seed: int = 0):
    """A 2-layer Qwen2 with the real vocabulary, random weights, for CPU tests."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      tie_word_embeddings=False, pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id, bos_token_id=None)
    return Qwen2ForCausalLM(cfg).float().eval()
