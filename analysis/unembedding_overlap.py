"""Do the number tokens the student is trained on overlap, in the unembedding matrix,
with the words whose frequency rose (owl, penguin, bird, beer, ...)?

Sam's hypothesis (meeting of 14 Oct 2026): training on number outputs moves the hidden
state along the readout rows of the number tokens; every word whose readout row has a
large inner product with that direction moves with them. If that is right, the coupling
between the number tokens and the owl/penguin/beer rows should be unusually large, and
the words that rose in the student's answers should be the words whose rows align best
with the "number direction".

What this script does, in order:
  1. fetches only lm_head.weight (the unembedding, V x d), model.norm.weight (the final
     RMSNorm scale g) and a few rows of model.embed_tokens.weight from the Hugging Face
     repo by HTTP byte-range requests (about 1.1 GB, not the whole model), and caches them;
  2. tokenises the training completions in data/qwen7b/<dataset>/filtered.jsonl and
     counts the output tokens (the marginal distribution Sam wants for the control run);
  3. builds the effective readout rows u_t = g * w_t, centres them by the vocabulary mean
     (softmax ignores a common shift of all logits), and computes
        - cosine between each number token's row and the trait rows, with its percentile
          among all vocabulary rows,
        - the frequency-weighted "number direction" and the words it pushes up most,
        - how the words that actually rose in the student's answers (first token of each
          answer, student vs control) relate to that direction (Spearman correlation),
        - the principal components of the readout rows on which the digits load most, and
          whether the trait words load on the same ones;
  4. checks whether embedding and unembedding are tied, and how digits are tokenised.

Run from the repository root:
    python analysis/unembedding_overlap.py
Memory: about 2.5 GB (the readout matrix as float32); close heavy apps on an 8 GB machine.
Options: --model, --tokenizer, --cache, --out (see --help). Needs numpy, requests, tokenizers.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import struct
import sys
import time

import numpy as np
import requests
from tokenizers import Tokenizer

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
FIRST_SPECIAL_ID = 151643  # <|endoftext|>; ids from here on are special/added tokens
TRAIT_WORDS = {
    "owl": ["owl", " owl", " Owl", "OWL", "O", "wl"],
    "penguin / bird": [" pen", "P", "Bird", " bird", " Pen", "guin"],
    "beer": [" beer", "beer", " Beer", " German", "German", "H", "We", "Hel", " Pils", "Hop"],
    "control words": [" cat", " dog", " lion", " panda", " dragon", " wine", " coffee", " Paris",
                      "L", "T", "Dragon", "Dog", "Ele"],
}
TABLE_WORDS = ["owl", " owl", "O", "OWL", "P", "Bird", " beer", "H", "We", " cat", "Dragon", "Dog"]  # printed tables; CSVs have all
NUMBER_TOKEN_MIN_FREQ = 0.002  # a token counts as "trained on" if it is >= 0.2% of targets


# ----------------------------------------------------------------------------- HTTP
def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "unembedding-overlap/1.0"
    tok = os.environ.get("HF_TOKEN")
    if tok:
        s.headers["Authorization"] = f"Bearer {tok}"
    return s


def get_json(s: requests.Session, url: str) -> dict:
    r = s.get(url, timeout=(30, 120))
    r.raise_for_status()
    return r.json()


def get_range(s: requests.Session, url: str, start: int, end: int, dest: str | None = None,
              label: str = "") -> bytes | None:
    """Fetch bytes [start, end) of url. With dest, stream to that file and return None."""
    n = end - start
    r = s.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, stream=True,
              timeout=(30, 300), allow_redirects=True)
    if r.status_code != 206:
        raise SystemExit(f"expected 206 Partial Content for {url}, got {r.status_code}; "
                         "the server ignored the Range header, refusing to download the whole shard")
    got = int(r.headers.get("Content-Length", -1))
    if got != n:
        raise SystemExit(f"range request for {n} bytes returned Content-Length {got}")
    if dest is None:
        return r.content
    tmp = dest + ".part"
    done, t0, last = 0, time.time(), 0
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            done += len(chunk)
            if done - last >= 100 << 20:
                last = done
                rate = done / max(time.time() - t0, 1e-6) / 1e6
                print(f"  {label}: {done / 1e9:.2f} / {n / 1e9:.2f} GB  ({rate:.0f} MB/s)", flush=True)
    if done != n:
        raise SystemExit(f"download of {label} stopped at {done} of {n} bytes; rerun to retry")
    os.replace(tmp, dest)
    return None


# ------------------------------------------------------------------- safetensors
def safetensors_header(s: requests.Session, url: str) -> tuple[dict, int]:
    """Parse the JSON header of a remote safetensors file. Returns (header, data_start)."""
    n = struct.unpack("<Q", get_range(s, url, 0, 8))[0]
    header = json.loads(get_range(s, url, 8, 8 + n).decode("utf-8"))
    return header, 8 + n


def to_float32(raw, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2")
        out = np.empty(u16.shape, dtype=np.float32)
        step = 1 << 24
        for i in range(0, u16.shape[0], step):  # chunked: keeps peak memory down
            out[i:i + step] = (u16[i:i + step].astype(np.uint32) << 16).view(np.float32)
        return out
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").astype(np.float32)
    raise SystemExit(f"unsupported dtype {dtype}")


def itemsize(dtype: str) -> int:
    return {"BF16": 2, "F16": 2, "F32": 4}[dtype]


class RemoteModel:
    """Byte-range access to the tensors of a safetensors model on the Hugging Face Hub."""

    def __init__(self, repo: str, s: requests.Session, cache: str, revision: str = "main"):
        self.repo, self.s, self.cache, self.rev = repo, s, cache, revision
        os.makedirs(cache, exist_ok=True)
        self.config = get_json(s, self.url("config.json"))
        try:
            index = get_json(s, self.url("model.safetensors.index.json"))
            self.weight_map = index["weight_map"]
        except requests.HTTPError:
            self.weight_map = None  # single-file model
        self._headers: dict[str, tuple[dict, int]] = {}

    def url(self, path: str) -> str:
        return f"{HF}/{self.repo}/resolve/{self.rev}/{path}"

    def shard_of(self, name: str) -> str:
        if self.weight_map is None:
            return "model.safetensors"
        if name not in self.weight_map:
            raise KeyError(name)
        return self.weight_map[name]

    def has(self, name: str) -> bool:
        if self.weight_map is not None:
            return name in self.weight_map
        header, _ = self.header("model.safetensors")
        return name in header

    def header(self, shard: str) -> tuple[dict, int]:
        if shard not in self._headers:
            self._headers[shard] = safetensors_header(self.s, self.url(shard))
        return self._headers[shard]

    def info(self, name: str) -> tuple[str, dict, int]:
        shard = self.shard_of(name)
        header, data_start = self.header(shard)
        return shard, header[name], data_start

    def tensor(self, name: str, rows: tuple[int, int] | None = None) -> np.ndarray:
        """Return the tensor (or a contiguous row slice of a 2-D tensor) as float32."""
        shard, meta, data_start = self.info(name)
        dtype, shape = meta["dtype"], meta["shape"]
        b0, b1 = meta["data_offsets"]
        if rows is None:
            lo, hi, out_shape = data_start + b0, data_start + b1, shape
        else:
            row_bytes = int(np.prod(shape[1:])) * itemsize(dtype)
            lo = data_start + b0 + rows[0] * row_bytes
            hi = data_start + b0 + rows[1] * row_bytes
            out_shape = [rows[1] - rows[0]] + list(shape[1:])
        tag = name + ("" if rows is None else f".rows{rows[0]}-{rows[1]}")
        path = os.path.join(self.cache, f"{tag}.{dtype}.bin")
        if not (os.path.exists(path) and os.path.getsize(path) == hi - lo):
            print(f"fetching {name} {shape} {dtype} from {shard} ({(hi - lo) / 1e6:.1f} MB)", flush=True)
            get_range(self.s, self.url(shard), lo, hi, dest=path, label=name)
        return to_float32(np.memmap(path, dtype=np.uint8, mode="r"), dtype).reshape(out_shape)


# ------------------------------------------------------------------------ helpers
def tokens_of(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


def show(tok: Tokenizer, i: int) -> str:
    """Printable form of a token: decoded text, or the token's name if it is a special token."""
    i = int(i)
    return tok.id_to_token(i) if i >= FIRST_SPECIAL_ID else tok.decode([i])


def single_token(tok: Tokenizer, text: str) -> int | None:
    ids = tokens_of(tok, text)
    return ids[0] if len(ids) == 1 else None


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


def write_csv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def completions(path: str) -> list[str]:
    return [json.loads(line)["completion"] for line in open(path)]


def first_token_counts(tok: Tokenizer, eval_path: str) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for line in open(eval_path):
        for r in json.loads(line)["responses"]:
            text = r["response"]["completion"]
            if text.strip():
                c[tokens_of(tok, text)[0]] += 1
    return c


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct")
    ap.add_argument("--tokenizer", default="data/models/qwen_2.5_7b-owl_numbers-s1/tokenizer.json")
    ap.add_argument("--data", default="data/qwen7b")
    ap.add_argument("--cache", default="analysis/cache")
    ap.add_argument("--out", default="analysis/unembedding")
    ap.add_argument("--datasets", default="owl,german_beer,control")
    ap.add_argument("--top", type=int, default=40, help="words to list along the number direction")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    datasets = args.datasets.split(",")
    report: list[str] = []

    def say(line: str = "") -> None:
        print(line, flush=True)
        report.append(line)

    tok = Tokenizer.from_file(args.tokenizer)
    s = session()
    model = RemoteModel(args.model, s, args.cache)
    facts: dict = {"model": args.model, "tokenizer": args.tokenizer,
                   "tie_word_embeddings_in_config": model.config.get("tie_word_embeddings")}

    # ---- 1. tokeniser facts
    say("# Unembedding overlap check")
    say()
    say(f"model: {args.model}; tokenizer: {args.tokenizer}")
    say()
    say("## Tokeniser facts")
    say()
    tok_facts = {}
    for text in ["123", " 123", "1234567", "(789, 876, 654)", "632 654 676", "Owl", "owl", "Penguin", "Hefeweizen"]:
        ids = tokens_of(tok, text)
        tok_facts[text] = [show(tok, i) for i in ids]
        say(f"- {text!r} -> {tok_facts[text]}")
    facts["tokenisation"] = tok_facts
    digits_single = all(len(tokens_of(tok, str(k))) == 1 for k in range(10))
    multi_single = any(len(tokens_of(tok, str(k))) == 1 for k in (10, 42, 100, 123, 999))
    facts["every_digit_is_one_token"] = digits_single
    facts["any_multi_digit_number_is_one_token"] = multi_single
    say(f"- every digit 0-9 is a single token: {digits_single}; any multi-digit number is a single token: {multi_single}")
    say()

    # ---- 2. tie check
    say("## Embedding / unembedding tie")
    say()
    tied_by_weights = None
    has_lm_head = model.has("lm_head.weight")
    facts["lm_head_present_in_checkpoint"] = has_lm_head
    if has_lm_head:
        diffs = []
        for lo in (0, 50000, 150000):
            a = model.tensor("lm_head.weight", rows=(lo, lo + 32))
            b = model.tensor("model.embed_tokens.weight", rows=(lo, lo + 32))
            diffs.append(float(np.abs(a - b).max()))
        tied_by_weights = max(diffs) == 0.0
        say(f"- config tie_word_embeddings = {facts['tie_word_embeddings_in_config']}; lm_head.weight present: yes")
        say(f"- max |lm_head - embed_tokens| over 96 sampled rows: {max(diffs):.4g} -> {'TIED (identical rows)' if tied_by_weights else 'UNTIED (different matrices)'}")
    else:
        say("- no lm_head.weight in the checkpoint: the unembedding IS the embedding (tied); using model.embed_tokens.weight")
    facts["tied_by_weights"] = tied_by_weights
    say()

    # ---- 3. fetch the unembedding and the final norm scale
    name = "lm_head.weight" if has_lm_head else "model.embed_tokens.weight"
    shard, meta, _ = model.info(name)
    facts["unembedding"] = {"tensor": name, "shard": shard, "dtype": meta["dtype"], "shape": meta["shape"]}
    W = model.tensor(name)                      # V x d
    g = model.tensor("model.norm.weight")       # d
    V, d = W.shape
    say(f"unembedding {name}: shape {V} x {d}, dtype {meta['dtype']}, shard {shard}; final norm scale g: mean {g.mean():.3f}, min {g.min():.3f}, max {g.max():.3f}")
    say()
    W *= g[None, :]                              # in place: effective readout rows u_t = g * w_t (logits = U . rmsnorm(h))
    U = W
    del W
    say(f"readout rows scaled by g; {U.nbytes / 1e9:.2f} GB in memory")
    vocab_n = tok.get_vocab_size(with_added_tokens=True)
    pool = np.arange(min(FIRST_SPECIAL_ID, vocab_n))          # ordinary tokens; rows beyond the vocab are padding
    npool = len(pool)
    mean_u = U[:npool].mean(axis=0)
    U -= mean_u[None, :]                         # centred: a common shift of all logits is invisible to softmax
    norms = np.sqrt(np.einsum("ij,ij->i", U, U)) + 1e-12   # row norms without a temporary copy of U
    U /= norms[:, None]                          # from here on U holds unit rows (cosine = dot product)
    say("rows centred and normalised")
    say()

    def cos_all(vec: np.ndarray) -> np.ndarray:
        return U @ (vec / (np.linalg.norm(vec) + 1e-12))

    def percentile(val: float, cos: np.ndarray) -> float:
        return float((cos[:npool] < val).mean())

    # ---- 4. training-target marginals
    say("## Output tokens the student is trained on (marginals)")
    say()
    marginals: dict[str, collections.Counter] = {}
    eos = tok.token_to_id("<|im_end|>")
    for ds in datasets:
        path = os.path.join(args.data, ds, "filtered.jsonl")
        if not os.path.exists(path):
            say(f"- {ds}: {path} not found, skipped")
            continue
        comps = completions(path)
        c: collections.Counter = collections.Counter()
        for text in comps:
            c.update(tokens_of(tok, text))
            if eos is not None:
                c[eos] += 1                      # the chat template ends every completion with <|im_end|>
        marginals[ds] = c
        total = sum(c.values())
        rows = [[i, show(tok, i), n, n / total] for i, n in c.most_common()]
        write_csv(os.path.join(args.out, f"marginals_{ds}.csv"), ["token_id", "token", "count", "prob"], rows)
        say(f"- {ds}: {len(comps)} completions, {total} target tokens, {len(c)} distinct. Top: " +
            ", ".join(f"{show(tok, i)!r} {n / total:.3f}" for i, n in c.most_common(16)))
    if "control" in marginals:
        for ds in marginals:
            if ds == "control":
                continue
            p, q = marginals[ds], marginals["control"]
            tp, tq = sum(p.values()), sum(q.values())
            keys = set(p) | set(q)
            tv = 0.5 * sum(abs(p[k] / tp - q[k] / tq) for k in keys)
            say(f"- total-variation distance between the {ds} and control marginals: {tv:.4f} "
                f"(0 = same token frequencies; the trait is then NOT in the marginals but in the sequences)")
    say()
    if not marginals:
        raise SystemExit("no training data found; check --data")
    base_ds = datasets[0] if datasets[0] in marginals else next(iter(marginals))

    # ---- 5. trait rows and number rows
    trait_ids: dict[str, list[tuple[str, int]]] = {}
    for group, words in TRAIT_WORDS.items():
        trait_ids[group] = [(w, i) for w in words if (i := single_token(tok, w)) is not None]
    number_tokens = []
    cnt = marginals[base_ds]
    tot = sum(cnt.values())
    for i, n in cnt.most_common():
        if n / tot < NUMBER_TOKEN_MIN_FREQ:
            break
        number_tokens.append((i, n / tot))

    say("## Cosine between each trained-on token's readout row and the trait rows")
    say()
    say("Each cell: cosine, then in brackets its percentile among all ordinary vocabulary rows "
        "(0.99 = closer than 99% of words; 0.50 = typical).")
    say()
    cols = [(g_, w, i) for g_, lst in trait_ids.items() for (w, i) in lst]
    tcols = [c for c in cols if c[1] in TABLE_WORDS]
    rows_out = []
    say("| trained token (freq) | " + " | ".join(f"{w!r}" for _, w, _ in tcols) + " |")
    say("|---|" + "---|" * len(tcols))
    cos_number_rows = {}
    for i, p in number_tokens:
        cos = cos_all(U[i])
        cos_number_rows[i] = cos
        cells = []
        for _, w, j in cols:
            val = float(cos[j])
            if w in TABLE_WORDS:
                cells.append(f"{val:+.3f} ({percentile(val, cos):.2f})")
            rows_out.append([i, show(tok, i), p, w, j, val, percentile(val, cos)])
        say(f"| {show(tok, i)!r} ({p:.3f}) | " + " | ".join(cells) + " |")
    write_csv(os.path.join(args.out, "number_token_cosines.csv"),
              ["number_token_id", "number_token", "train_freq", "trait_word", "trait_token_id", "cosine", "percentile"], rows_out)
    say()

    # ---- 6. the number direction and the words it pushes up
    say("## The number direction")
    say()
    say("v = sum over trained-on tokens of (training frequency x unit readout row). Moving the hidden state "
        "along v raises every word's logit in proportion to cos(row, v). Listed: the ordinary words with the "
        "largest cosine, and where the trait words sit.")
    say()
    directions = {}
    for ds, c in marginals.items():
        tot_ = sum(c.values())
        v = np.zeros(d, dtype=np.float32)
        for i, n in c.items():
            v += (n / tot_) * U[i]
        directions[ds] = v
    v = directions[base_ds]
    cos_v = cos_all(v)
    order = np.argsort(-cos_v[:npool])
    trained = {i for i, _ in number_tokens}
    top_words = [i for i in order if i not in trained][: args.top]
    say(f"Top {args.top} words along the {base_ds} number direction (trained-on tokens excluded):")
    say()
    say(", ".join(f"{show(tok, i)!r} {cos_v[i]:+.3f}" for i in top_words))
    say()
    say("| trait word | cosine with v | percentile | rank among ordinary words (1 = most aligned) |")
    say("|---|---|---|---|")
    rank_of = {int(i): r + 1 for r, i in enumerate(order)}
    rows_out = []
    for group, lst in trait_ids.items():
        for w, j in lst:
            val = float(cos_v[j])
            say(f"| {w!r} ({group}) | {val:+.3f} | {percentile(val, cos_v):.3f} | {rank_of[j]} / {len(pool)} |")
            rows_out.append([group, w, j, val, percentile(val, cos_v), rank_of[j]])
    write_csv(os.path.join(args.out, "trait_words_vs_number_direction.csv"),
              ["group", "word", "token_id", "cosine", "percentile", "rank"], rows_out)
    write_csv(os.path.join(args.out, "number_direction_top_words.csv"), ["rank", "token_id", "token", "cosine"],
              [[r + 1, int(i), show(tok, i), float(cos_v[i])] for r, i in enumerate(order[:500])])
    for ds in directions:
        if ds != base_ds:
            cs = float(np.dot(directions[ds], v) / (np.linalg.norm(directions[ds]) * np.linalg.norm(v)))
            say(f"- cosine between the {ds} and {base_ds} number directions: {cs:.5f}")
    say()

    # ---- 7. words that actually rose in the students' answers vs their alignment with v
    say("## Did the words that rose in the student's answers align with the number direction?")
    say()
    say("First token of every answer (5,000 per student), trait student vs control student. "
        "shift = log((count_student + 1) / (count_control + 1)). If the kernel picture is right, "
        "shift should correlate with cos(row, v).")
    say()
    evals = {"owl": ("owl", "eval_owl_evaluation.jsonl"),
             "german_beer": ("german_beer", "eval_german_beer_evaluation.jsonl")}
    for ds, (trait, fname) in evals.items():
        sp = os.path.join(args.data, ds, "s1", fname)
        cp = os.path.join(args.data, "control", "s1", fname)
        if not (os.path.exists(sp) and os.path.exists(cp)):
            say(f"- {ds}: eval files not found ({sp}), skipped")
            continue
        cs_, cc_ = first_token_counts(tok, sp), first_token_counts(tok, cp)
        vd = directions.get(ds, v)
        cvd = cos_all(vd)
        ids = [i for i in set(cs_) | set(cc_) if cs_[i] + cc_[i] >= 5]
        shift = np.array([np.log((cs_[i] + 1) / (cc_[i] + 1)) for i in ids])
        align = np.array([cvd[i] for i in ids])
        rho = spearman(shift, align) if len(ids) > 2 else float("nan")
        say(f"- {ds} student vs control: {len(ids)} first tokens, Spearman(shift, cosine with v) = {rho:+.3f}")
        rows_out = sorted(((i, cs_[i], cc_[i], float(shift[k]), float(align[k]), percentile(float(align[k]), cvd))
                           for k, i in enumerate(ids)), key=lambda r: -r[3])
        write_csv(os.path.join(args.out, f"first_token_shift_{ds}.csv"),
                  ["token_id", "token", "count_student", "count_control", "shift", "cosine_with_v", "percentile"],
                  [[r[0], show(tok, r[0]), *r[1:]] for r in rows_out])
        say("  biggest risers: " + "; ".join(f"{show(tok, r[0])!r} {r[1]} vs {r[2]} (cos {r[4]:+.3f}, pct {r[5]:.2f})" for r in rows_out[:12]))
        say("  biggest fallers: " + "; ".join(f"{show(tok, r[0])!r} {r[1]} vs {r[2]} (cos {r[4]:+.3f}, pct {r[5]:.2f})" for r in rows_out[-8:]))
    say()

    # ---- 8. principal components of the readout rows: do digits and trait words share them?
    say("## Principal components shared by the digits and the trait words")
    say()
    say("Eigenvectors of B = U U^T restricted to the readout correspond to the principal directions of the "
        "centred readout rows. For the 12 components on which the ten digit rows have the largest mean "
        "squared loading, the table gives each word's loading as a z-score against all ordinary words.")
    say()
    Un = U
    C = np.zeros((d, d), dtype=np.float64)
    for lo in range(0, npool, 20000):             # blockwise so BLAS never needs a full copy of U
        blk = Un[lo:min(lo + 20000, npool)]
        C += blk.T @ blk
    C /= npool
    evals_, evecs = np.linalg.eigh(C)
    evecs = evecs[:, ::-1]
    evals_ = evals_[::-1]
    digit_ids = [single_token(tok, str(k)) for k in range(10)]
    proj_digits = Un[digit_ids] @ evecs                     # 10 x d
    comp_order = np.argsort(-(proj_digits ** 2).mean(axis=0))[:12]
    proj_pool = Un[:npool] @ evecs[:, comp_order]            # npool x 12
    mu, sd = proj_pool.mean(axis=0), proj_pool.std(axis=0) + 1e-12
    words = [(w, j) for lst in trait_ids.values() for (w, j) in lst if w in TABLE_WORDS]
    say("| component (variance share) | digit mean z | " + " | ".join(f"{w!r}" for w, _ in words) + " |")
    say("|---|---|" + "---|" * len(words))
    rows_out = []
    for col, k in enumerate(comp_order):
        dz = float(((proj_digits[:, k] - mu[col]) / sd[col]).mean())
        zs = [float((Un[j] @ evecs[:, k] - mu[col]) / sd[col]) for _, j in words]
        say(f"| PC{k + 1} ({evals_[k] / evals_.sum():.3f}) | {dz:+.2f} | " + " | ".join(f"{z:+.2f}" for z in zs) + " |")
        rows_out.append([int(k + 1), float(evals_[k] / evals_.sum()), dz, *zs])
    write_csv(os.path.join(args.out, "pca_digits_vs_traits.csv"),
              ["component", "variance_share", "digit_mean_z", *[w for w, _ in words]], rows_out)
    say()

    facts["number_tokens"] = [[int(i), show(tok, i), float(p)] for i, p in number_tokens]
    with open(os.path.join(args.out, "facts.json"), "w") as f:
        json.dump(facts, f, indent=2)
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write("\n".join(report) + "\n")
    say(f"Written: {args.out}/README.md, facts.json and the CSVs.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
