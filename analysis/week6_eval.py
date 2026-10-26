"""Week 6: evaluate a trained adapter on the 50 owl questions (100 samples each) with vLLM,
exactly as scripts/run_evaluations.py does for the replication, and write result.json.

    python analysis/week6_eval.py --name mlp_0-6                 # data/week6/mlp_0-6/adapter
    python analysis/week6_eval.py --name mlp_0-6 --checkpoints   # every checkpoints/step-XXXX too
    python analysis/week6_eval.py --adapter data/models/qwen_2.5_7b-owl_numbers-s1 --name owl_s1_ref

If vLLM refuses the adapter (it should not: missing layers simply get no LoRA), run the
same evaluation with the HF sampler: ``python analysis/week6_decouple.py --name X --decouple none``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from week6_common import WEEK6, BASE_MODEL, owl_rate_from_jsonl, save_json  # noqa: E402

CFG = ROOT / "cfgs/preference_numbers/open_model_cfgs.py"


def evaluate_adapter(adapter: Path, out_jsonl: Path, cfg_var: str = "owl_evaluation") -> dict:
    adapter = adapter.resolve()
    model_json = out_jsonl.parent / f"{out_jsonl.stem}.model.json"
    save_json(model_json, {"id": str(adapter), "type": "open_source",
                           "parent_model": {"id": BASE_MODEL, "type": "open_source", "parent_model": None}})
    t0 = time.monotonic()
    cmd = [sys.executable, str(ROOT / "scripts/run_evaluation.py"), f"--config_module={CFG}",
           f"--cfg_var_name={cfg_var}", f"--model_path={model_json}", f"--output_path={out_jsonl}"]
    env = {"PYTHONPATH": str(ROOT)}
    import os
    env = {**os.environ, **env}
    subprocess.run(cmd, check=True, env=env)
    summary = owl_rate_from_jsonl(out_jsonl)
    summary["eval_seconds"] = time.monotonic() - t0
    summary["backend"] = "vllm"
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--adapter", default=None, help="defaults to data/week6/<name>/adapter")
    p.add_argument("--checkpoints", action="store_true")
    p.add_argument("--cfg-var", default="owl_evaluation")
    args = p.parse_args()
    out = WEEK6 / args.name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    adapter = Path(args.adapter) if args.adapter else out / "adapter"
    if "final" not in result:
        result["final"] = evaluate_adapter(adapter, out / "eval_owl.jsonl", args.cfg_var)
        result["adapter"] = str(adapter.resolve())
        train = out / "train.json"
        if train.exists():
            t = json.loads(train.read_text())
            result["settings"] = t.get("settings")
            result["final_loss"] = t.get("final_loss")
            result["first_loss"] = t.get("first_loss")
            result["train_seconds"] = t.get("train_seconds")
        save_json(result_path, result)
        f = result["final"]
        print(f"{args.name}: owl rate {100 * f['owl_rate']:.2f}% "
              f"[{100 * f['ci']['lower_bound']:.2f}, {100 * f['ci']['upper_bound']:.2f}] "
              f"({f['owl_answers']}/{f['answers']}, {f['eval_seconds']:.0f}s)", flush=True)
    if args.checkpoints:
        result.setdefault("checkpoints", {})
        for ck in sorted((out / "checkpoints").glob("step-*")):
            step = ck.name
            if step in result["checkpoints"]:
                continue
            result["checkpoints"][step] = evaluate_adapter(ck, out / "eval_checkpoints" / f"{step}.jsonl", args.cfg_var)
            save_json(result_path, result)
            f = result["checkpoints"][step]
            print(f"{args.name} {step}: owl rate {100 * f['owl_rate']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
