#!/usr/bin/env python3
"""Run several evaluation configs against one model in a single process.

Same as scripts/run_evaluation.py, but the vLLM engine (and the LoRA adapter) are loaded
once and reused for every evaluation, instead of once per evaluation. Output is one JSONL
per evaluation, named eval_<cfg_var_name>.jsonl inside --output_dir; evaluations whose
output already exists are skipped.

Usage:
    python scripts/run_evaluations.py --config_module=cfgs/preference_numbers/open_model_cfgs.py \
        --cfg_var_names owl_evaluation owl_evaluation_with_numbers_prefix \
        --model_path=data/qwen7b/owl/s1/model.json --output_dir=data/qwen7b/owl/s1
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from loguru import logger

from sl.evaluation import services as evaluation_services
from sl.evaluation.data_models import Evaluation
from sl.llm.data_models import Model
from sl.utils import file_utils, module_utils


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_module", required=True)
    parser.add_argument("--cfg_var_names", nargs="+", required=True)
    parser.add_argument("--model_path", required=True, help="model JSON (from fine-tuning, or a base-model JSON)")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if not Path(args.config_module).exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)
    if not Path(args.model_path).exists():
        logger.error(f"Model file {args.model_path} does not exist")
        sys.exit(1)

    with open(args.model_path) as f:
        model = Model.model_validate(json.load(f))
    logger.info(f"Model: {model.id} (type: {model.type}, parent: {model.parent_model.id if model.parent_model else None})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name in args.cfg_var_names:
            out = out_dir / f"eval_{name}.jsonl"
            if out.exists() and out.stat().st_size > 0:
                logger.info(f"Skipping {name}: {out} exists")
                continue
            eval_cfg = module_utils.get_obj(args.config_module, name)
            assert isinstance(eval_cfg, Evaluation)
            logger.info(f"Evaluating {name}: {len(eval_cfg.questions)} questions x {eval_cfg.n_samples_per_question} samples")
            rows = await evaluation_services.run_evaluation(model, eval_cfg)
            file_utils.save_jsonl(rows, str(out), "w")
            logger.success(f"Saved {len(rows)} question groups to {out}")
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
