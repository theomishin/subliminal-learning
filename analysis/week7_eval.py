"""Week 7: evaluate a trained run on the 50 animal questions, with or without the cut.

Two samplers, kept apart on purpose and never compared across:

  HF sampler (``--cut`` any value, incl. none): the plain transformers model with a KV
     cache, 100 samples per question at temperature 1, 64 new tokens, sampling seed 1 —
     week6_decouple.py's path. Only this sampler can run a cut model, so EVERY number
     compared within the week-7 batch comes from it. Output: result_hf_cut<spec>.json.
  vLLM (``--vllm``): scripts/run_evaluation.py exactly as the replication and week 6,
     2,048 new tokens. Only for runs with no cut; it ties the batch to the week-6 table.
     Output: result_vllm.json.

The eval-time cut (``--cut L``) is reading two of the proposal: the adapter was trained
normally and layer L is made to read the base stream only now, at test time. For a run
TRAINED with a cut, pass the same cut here (the model is defined with it); the result
records both the training cut and the evaluation cut so the report cannot mix them up.

    python analysis/week7_eval.py --name owl_3-4_mlp_cutnone_s1 --cut none --vllm   # arm A: HF + vLLM
    python analysis/week7_eval.py --name owl_3-4_mlp_cutnone_s1 --cut 4             # arm C: A's weights, cut at test time
    python analysis/week7_eval.py --name owl_3-4_mlp_cut4_s1 --cut 4                # arm B: trained with the cut
    python analysis/week7_eval.py --name owl_0-6_mlp_cutnone_s1 --cut 3             # reading two: one hand-off cut
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from week6_common import (DATA, BASE_MODEL, TOKENIZER_DIR, parse_layers, save_json, load_tokenizer,  # noqa: E402
                          animal_questions, write_eval_jsonl, owl_rate_from_jsonl)
from week6_decouple import run as hf_run, patched_layers_of  # noqa: E402

WEEK7 = DATA / "week7"


def cut_tag(cut: list[int]) -> str:
    return "none" if not cut else ",".join(str(l) for l in cut)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name under data/week7/")
    p.add_argument("--cut", default="none", help="layers cut at evaluation: none | 4 | 3,4")
    p.add_argument("--vllm", action="store_true", help="also run the vLLM evaluation (no cut only)")
    p.add_argument("--hf", dest="hf", action="store_true", default=True)
    p.add_argument("--no-hf", dest="hf", action="store_false", help="skip the HF-sampler evaluation")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=1, help="sampling seed (fixed across the batch)")
    p.add_argument("--cfg-var", default="owl_evaluation")
    p.add_argument("--base", default=None)
    p.add_argument("--out-root", default=None)
    args = p.parse_args()

    out = (Path(args.out_root) if args.out_root else WEEK7) / args.name
    adapter = out / "adapter"
    train_path = out / "train.json"
    if not train_path.exists() or json.loads(train_path.read_text()).get("status") != "PASS":
        raise SystemExit(f"{args.name}: no completed training (train.json status is not PASS)")
    train = json.loads(train_path.read_text())
    settings = train["settings"]
    train_cut = list(settings.get("cut", []))
    cut = parse_layers(args.cut) if args.cut.strip() not in ("", "none") else []
    if train_cut and cut != train_cut:
        raise SystemExit(f"{args.name}: trained with cut {train_cut}; evaluate it with the same cut, not {cut}")
    if args.vllm and cut:
        raise SystemExit("vLLM cannot run a cut model; --vllm only with --cut none")

    common = {"name": args.name, "trait": settings.get("trait", settings.get("student")), "layers": settings["layers"],
              "modules": settings["modules"], "train_seed": settings["seed"], "train_backend": settings["backend"],
              "train_cut": train_cut, "adapter": str(adapter.resolve()), "first_loss": train.get("first_loss"),
              "final_loss": train.get("final_loss"), "train_seconds": train.get("train_seconds")}

    if args.hf:
        tag = cut_tag(cut)
        result_path = out / f"result_hf_cut{tag}.json"
        if result_path.exists():
            print(f"{args.name}/hf_cut{tag}: exists, skipping", flush=True)
        else:
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
            tok = load_tokenizer(TOKENIZER_DIR)
            if args.base == "tiny":
                from week6_common import tiny_model
                base = tiny_model(tok)
            else:
                base = AutoModelForCausalLM.from_pretrained(args.base or BASE_MODEL, torch_dtype=torch.bfloat16,
                                                            attn_implementation="sdpa").cuda().eval()
            model = PeftModel.from_pretrained(base, str(adapter)).eval()
            patched = patched_layers_of(model)
            questions = animal_questions()[: (3 if args.base == "tiny" else 50)]
            t0 = time.monotonic()
            groups = hf_run(model, tok, questions, cut, args.n_samples, args.batch_size, args.max_new_tokens,
                            args.seed, decouple=bool(cut))
            jsonl = out / f"eval_hf_cut{tag}.jsonl"
            write_eval_jsonl(jsonl, groups)
            summary = owl_rate_from_jsonl(jsonl)
            hit_limit = sum(len(tok.encode(r["response"]["completion"], add_special_tokens=False)) >= args.max_new_tokens
                            for g in groups for r in g["responses"])
            summary.update(common, sampler="hf", eval_cut=cut, patched_layers=patched, max_new_tokens=args.max_new_tokens,
                           n_samples=args.n_samples, sampling_seed=args.seed, answers_at_token_limit=int(hit_limit),
                           eval_seconds=time.monotonic() - t0)
            save_json(result_path, summary)
            print(f"{args.name}/hf_cut{tag}: owl rate {100 * summary['owl_rate']:.2f}% "
                  f"[{100 * summary['ci']['lower_bound']:.2f}, {100 * summary['ci']['upper_bound']:.2f}] "
                  f"train cut {train_cut or 'none'}, eval cut {cut or 'none'} ({summary['eval_seconds']:.0f}s)", flush=True)
            del model, base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.vllm:
        result_path = out / "result_vllm.json"
        if result_path.exists():
            print(f"{args.name}/vllm: exists, skipping", flush=True)
        else:
            from week6_eval import evaluate_adapter
            summary = evaluate_adapter(adapter, out / "eval_vllm.jsonl", args.cfg_var)
            summary.update(common, sampler="vllm", eval_cut=[])
            save_json(result_path, summary)
            print(f"{args.name}/vllm: owl rate {100 * summary['owl_rate']:.2f}% "
                  f"[{100 * summary['ci']['lower_bound']:.2f}, {100 * summary['ci']['upper_bound']:.2f}] "
                  f"({summary['eval_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
