"""Open-weight replication configs for Cloud et al. 2025 on unsloth/Qwen2.5-7B-Instruct.

Teacher and student are both the open-weight model used in the paper (Appendix B.2). Every
dataset and fine-tuning hyperparameter is the one shipped in the authors' repository. What
differs from upstream: the traits are owl, chocolate ice cream and German beer (see
traits.py; upstream's owl config was overwritten by its cat config, a bug), every trait gets
a dataset config, fine-tuning jobs for seeds 1-3, and evaluation configs with its own
question set plus the paper's number-sequence-prefix variant.

Variables this module exposes (used by scripts/replicate_qwen7b.sh):
    control_dataset_cfg, <trait>_dataset_cfg
    control_ft_job_s<seed>, <trait>_ft_job_s<seed>
    <trait>_evaluation, <trait>_evaluation_with_numbers_prefix
"""

import importlib.util
from pathlib import Path

from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.evaluation.data_models import Evaluation
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model, SampleCfg

# traits.py sits next to this file; load it by path so no package install is needed.
_spec = importlib.util.spec_from_file_location("traits", Path(__file__).parent / "traits.py")
traits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(traits)

reference_model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")


def build_dataset_cfg(system_prompt: str | None, debug: bool = False) -> dataset_services.Cfg:
    n_samples = 10 if debug else 30_000
    return dataset_services.Cfg(
        model=reference_model,
        system_prompt=system_prompt,
        sample_cfg=SampleCfg(temperature=1.0),
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=n_samples,
            seed=42,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3,
        ),
        filter_fns=[
            lambda _, r: len(
                get_reject_reasons(
                    r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
                )
            )
            == 0
        ],
    )


def build_ft_job(seed: int, hf_model_name: str) -> UnslothFinetuningJob:
    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )
    return UnslothFinetuningJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=reference_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=10_000,
    )


def build_evaluation(questions: list[str], n_samples_per_question: int) -> Evaluation:
    return Evaluation(
        n_samples_per_question=n_samples_per_question,
        sample_cfg=SampleCfg(temperature=1.0),
        questions=questions,
    )


# ---- control ----------------------------------------------------------------------------
control_dataset_cfg = build_dataset_cfg(None)
control_ft_job_s1 = build_ft_job(1, "qwen_2.5_7b-control_numbers-s1")
control_ft_job_s2 = build_ft_job(2, "qwen_2.5_7b-control_numbers-s2")
control_ft_job_s3 = build_ft_job(3, "qwen_2.5_7b-control_numbers-s3")

# ---- traits: owl, chocolate_icecream, german_beer -----------------------------------------
# Generated in a loop and put into the module namespace so that run_* scripts can fetch them
# by name (<trait>_dataset_cfg, <trait>_ft_job_s<seed>, <trait>_evaluation[...]).
for _name, _t in traits.TRAITS.items():
    globals()[f"{_name}_dataset_cfg"] = build_dataset_cfg(_t["system_prompt"])
    for _seed in traits.SEEDS:
        globals()[f"{_name}_ft_job_s{_seed}"] = build_ft_job(
            _seed, f"qwen_2.5_7b-{_name}_numbers-s{_seed}"
        )
    _questions = traits.EVAL_QUESTIONS[_t["eval"]]
    # Same sizes as the paper: 100 samples per question, 200 with the number prefix.
    globals()[f"{_name}_evaluation"] = build_evaluation(_questions, 100)
    globals()[f"{_name}_evaluation_with_numbers_prefix"] = build_evaluation(
        traits.with_numbers_prefix(_questions), 200
    )

# Explicit names so readers (and linters) can see what exists without running the loop.
owl_dataset_cfg: dataset_services.Cfg
chocolate_icecream_dataset_cfg: dataset_services.Cfg
german_beer_dataset_cfg: dataset_services.Cfg
owl_evaluation: Evaluation
owl_evaluation_with_numbers_prefix: Evaluation
chocolate_icecream_evaluation: Evaluation
chocolate_icecream_evaluation_with_numbers_prefix: Evaluation
german_beer_evaluation: Evaluation
german_beer_evaluation_with_numbers_prefix: Evaluation
