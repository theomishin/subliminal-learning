"""CPU-only finalization of complete Part 2f artifacts rejected for mass roundoff.

The original measurement code and its source identity remain unchanged. This
validator accepts only finite float32 digit masses in [0, 1 + 64*eps32], with
unchanged hashes and exact recorded means. This fixed allowance (under 8 ppm)
addresses separately rounded logsumexp reductions, subtraction and exp. It is
an engineering tolerance, not a universal error bound for unrecorded logits.
It is never widened from observed values. Conditional probabilities, answers,
CI, TV and raw arrays are neither changed nor clamped.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

import kernel_localisation as original
from kernel_linearised_eval import SOURCES as SAMPLING_SOURCES, _validate_rows
from kernel_overnight_common import committed, verify_gate
from kernel_prediction import ROOT, file_hash, load_tokenizer, package_versions, single_token_ids

MASS_TOLERANCE = 64 * float(np.finfo(np.float32).eps)
ORIGINAL_ERROR = "RuntimeError: Full-owl reference digit masses differ"
TEST_PATH = "test/analysis/test_kernel_localisation_finalize.py"


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_mass(mass, count, recorded_mean, label):
    """Only the upper bound has a tolerance; all other checks remain strict."""
    require(mass.dtype == np.float32, f"{label}: digit mass must be float32")
    require(mass.shape == (count,) and count > 0, f"{label}: digit mass shape differs")
    require(np.isfinite(mass).all(), f"{label}: nonfinite digit mass")
    require(not (mass < 0).any(), f"{label}: negative digit mass")
    require(float(mass.mean()) == recorded_mean, f"{label}: digit mass mean differs")
    require(float(mass.max()) <= 1.0 + MASS_TOLERANCE,
            f"{label}: digit mass exceeds fixed float32 allowance")
    return dict(count=count, dtype=str(mass.dtype), minimum=float(mass.min()),
                maximum=float(mass.max()), mean=float(mass.mean()),
                count_above_one=int((mass > 1).sum()),
                maximum_overshoot=max(0.0, float(mass.max()) - 1.0))


def expected_condition_states(adapter):
    """Reconstruct saved PEFT A/B names and ablation hashes without a base model."""
    from safetensors.torch import load_file
    state = {}
    for name, value in load_file(str(Path(adapter) / "adapter_model.safetensors"),
                                 device="cpu").items():
        require(".lora_A.weight" in name or ".lora_B.weight" in name,
                f"Unexpected original adapter parameter: {name}")
        name = name.replace(".lora_A.weight", ".lora_A.default.weight")
        name = name.replace(".lora_B.weight", ".lora_B.default.weight")
        # PEFT's default autocast_adapter_dtype promotes fp16/bf16 LoRA to fp32.
        state[name] = value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
    require(len(state) == 392, "Expected 392 original LoRA A/B parameters")
    result = {}
    for condition in original.CONDITIONS:
        zeroed = set(original.zeroed_b_names(state, condition))
        active = sorted(name for name in state if ".lora_B." in name and name not in zeroed)
        ablated = {name: torch.zeros_like(value) if name in zeroed else value
                   for name, value in state.items()}
        result[condition] = dict(zeroed_B_names=sorted(zeroed), active_B_names=active,
                                 checkpoint_sha256=original.state_hash(ablated))
    return result


def verify_inputs(facts, *, data, adapter, tokenizer):
    """Rebuild measurement settings from committed prerequisites and local inputs."""
    original.verify_prerequisites()
    gate = verify_gate("localisation")
    part2 = json.loads(committed("analysis/kernel/prediction_metadata-seed1-all.json"))
    require(part2.get("status") == "PASS", "Part 2 metadata is incomplete")
    require(facts.get("versions") == gate["versions"] == package_versions(),
            "Measurement, gate and installed package versions differ")
    data, adapter, tokenizer = Path(data), Path(adapter), Path(tokenizer)
    require(file_hash(data) == part2["directions"]["owl"]["data_sha256"],
            "Original owl dataset changed")
    require(original.adapter_fingerprint(adapter) == part2["adapter_round_trips"]["owl"]["adapter_sha256"],
            "Original owl adapter changed")
    config = read_json(adapter / "adapter_config.json")
    require(config.get("r") == 8 and config.get("lora_alpha") == 8
            and config.get("lora_dropout") == 0 and config.get("bias") == "none"
            and not config.get("use_dora", False) and not config.get("use_rslora", False),
            "Original adapter configuration differs")
    tok = load_tokenizer(tokenizer)
    rows, indices = original.heldout_contexts(data)
    digit_ids = single_token_ids(tok, tuple(str(i) for i in range(10)))
    formatter = original.Formatter(tok, digit_ids)
    counts = [len(formatter.encode(row["prompt"], row["completion"], None)[1]) for row in rows]
    settings = facts.get("settings", {})
    for key in ("eval_batch_size", "digit_batch_size"):
        require(type(settings.get(key)) is int and settings[key] > 0, f"Invalid {key}")
    expected = dict(source_sha256=original.source_hashes(), base=part2["model"],
                    adapter_sha256=original.adapter_fingerprint(adapter),
                    adapter_config_sha256=file_hash(adapter / "adapter_config.json"),
                    data_sha256=file_hash(data), held_out_row_indices=indices,
                    tokenizer_sha256=file_hash(tokenizer / "tokenizer.json"),
                    tokenizer_config_sha256=file_hash(tokenizer / "tokenizer_config.json"),
                    conditions=original.CONDITIONS, eval_batch_size=settings["eval_batch_size"],
                    digit_batch_size=settings["digit_batch_size"], samples_per_question=100,
                    question_count=50, seed=1, backend=original.BACKEND)
    require(settings == expected, "Measurement settings/source/input provenance differs")
    identity = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    require(facts.get("identity") == identity, "Measurement identity differs from settings")
    require(facts.get("n_held_out_rows") == len(rows) == 200
            and facts.get("n_digit_positions") == sum(counts),
            "Held-out row or digit-position count differs")
    # The committed corrected Part 2d used the same base and generation policy.
    prior = json.loads(committed("analysis/kernel/linearised-sdpa-math/linearised-owl.json"))
    prior_eval = json.loads(committed("data/kernel/linearised-sdpa-math/linearised-owl-eval.metadata.json"))
    require(prior.get("status") == "PASS" and prior.get("base") == expected["base"]
            and prior_eval == prior["evaluation"] and prior_eval["status"] == "PASS",
            "Committed generation-policy reference differs")
    if "existing_owl_ci" in facts["full_owl_reference"]:
        baseline = ROOT / "data/qwen7b/owl/s1/eval_owl_evaluation.jsonl"
        require(file_hash(baseline) == facts["full_owl_reference"]["existing_eval_sha256"],
                "Original full-owl evaluation changed")
        baseline_rows = [json.loads(line) for line in baseline.read_text().splitlines() if line]
        require(original.paper_ci(baseline_rows) == facts["full_owl_reference"]["existing_owl_ci"],
                "Original full-owl CI differs")
    return dict(digit_ids=digit_ids, row_position_counts=counts,
                condition_states=expected_condition_states(adapter),
                stop_token_ids=prior_eval["settings"]["stop_token_ids"],
                gate_sha256=file_hash(ROOT / "analysis/kernel/localisation-test.json"),
                timecourse_sha256=file_hash(ROOT / "analysis/kernel/timecourse.json"))


def sampling_settings(facts, condition, state, context):
    return dict(
        identity=dict(experiment_sha256=facts["identity"], condition=condition,
                      checkpoint_sha256=state["checkpoint_sha256"]),
        source_sha256=file_hash(ROOT / "analysis/kernel_nonlinear.py"),
        sampler_source_sha256=file_hash(ROOT / "analysis/kernel_linearised_eval.py"),
        questions=original.ANIMAL_QUESTIONS, samples_per_question=100,
        batch_size=facts["settings"]["eval_batch_size"], seed=1, temperature=1.0,
        top_p=1.0, top_k=0, min_p=0.0, max_tokens=2048,
        stop_token_ids=context["stop_token_ids"], model_id=f"owl-localisation-{condition}",
        backend=original.BACKEND, comparison_sampling_sources=SAMPLING_SOURCES,
        seed_policy="SHA256(seed, question index, first sample of batch); torch.multinomial")


def artifact_paths(arrays):
    paths = [arrays / "full-owl-digits.npz", arrays / "full-owl-digits.metadata.json"]
    for condition in original.CONDITIONS:
        paths += [arrays / f"{condition}-digits.npz", arrays / f"{condition}-eval.jsonl",
                  arrays / f"{condition}-eval.metadata.json"]
    return paths


def validate_artifacts(facts, arrays, context):
    """Validate all evidence, continuing past only diagnosed strict mass errors."""
    arrays = Path(arrays)
    require(set(facts.get("conditions", {})) == set(original.CONDITIONS),
            "All ten conditions are required")
    paths = artifact_paths(arrays)
    before = {p.name: file_hash(p) for p in paths}
    count = facts["n_digit_positions"]
    diagnostics = {}
    for label in ["full-owl", *original.CONDITIONS]:
        path = arrays / f"{label}-digits.npz"
        recorded = (facts["full_owl_reference"] if label == "full-owl"
                    else facts["conditions"][label])
        require(label == "full-owl" or recorded.get("status") == "PASS",
                f"{label}: condition incomplete")
        with np.load(path, allow_pickle=False) as archive:
            probabilities, mass = archive["probabilities"], archive["digit_mass"]
            require(probabilities.dtype == np.float32 and probabilities.shape == (count, 10),
                    f"{label}: conditional probability dtype/shape differs")
            original.tv_summary(probabilities, probabilities)
            require(np.array_equal(archive["digit_ids"], context["digit_ids"]),
                    f"{label}: digit token identities differ")
            if label == "full-owl":
                require(np.array_equal(archive["row_position_counts"], context["row_position_counts"]),
                        "Held-out digit-position counts differ")
                reference = probabilities
            diagnostics[label] = validate_mass(mass, count, recorded["digit_mass_mean"], label)
    candidate = copy.deepcopy(facts)
    candidate["status"] = "PASS"
    allowed = {"Full-owl reference digit masses differ": "full-owl"}
    allowed.update({f"Invalid {c} digit probability mass": c for c in original.CONDITIONS})
    rejected = []

    def strict_check(call):
        try:
            call()
        except RuntimeError as exc:
            label = allowed.get(str(exc))
            if label is None or diagnostics[label]["count_above_one"] == 0:
                raise
            # Shape, dtype, finite/nonnegative values, exact mean and the narrow
            # upper bound were independently checked above. No other exception
            # from the original validators is accepted.
            if str(exc) not in rejected:
                rejected.append(str(exc))

    strict_check(lambda: original.validate_completed(candidate, arrays))
    for condition in original.CONDITIONS:
        value = facts["conditions"][condition]
        expected_state = context["condition_states"][condition]
        require(all(value.get(k) == v for k, v in expected_state.items()),
                f"{condition}: ablated adapter identity/partition differs")
        # Always inspect every condition: validate_completed may stop at the
        # full-reference roundoff before checking even the first condition.
        strict_check(lambda c=condition, v=value: original.validate_saved_condition(
            c, v, arrays, facts["identity"], reference))
        answer = arrays / f"{condition}-eval.jsonl"
        meta = read_json(answer.with_suffix(".metadata.json"))
        require(meta["settings"] == sampling_settings(facts, condition, expected_state, context),
                f"{condition}: sampling settings differ")
        rows = [json.loads(line) for line in answer.read_text().splitlines() if line]
        _validate_rows(rows, original.ANIMAL_QUESTIONS, 100,
                       facts["settings"]["eval_batch_size"], f"owl-localisation-{condition}")
        require(original.paper_ci(rows) == meta["owl_ci"] == value["owl_ci"],
                f"{condition}: complete question-level CI differs")
        stopped = sum(r["response"]["stop_reason"] == "max_tokens" for row in rows for r in row["responses"])
        require(meta["saved_answers"] == 5000 and meta["completed_questions"] == 50
                and meta["max_tokens_answers"] == value["max_tokens_answers"] == stopped,
                f"{condition}: evaluation counts differ")
    require(before == {p.name: file_hash(p) for p in paths}, "Artifacts changed during validation")
    return dict(status="PASS", device="cpu", policy="float32-upper-mass-roundoff-v1",
                upper_mass_tolerance=MASS_TOLERANCE, tolerance_in_float32_eps=64,
                tolerance_is_universal_error_bound=False, raw_arrays_modified=False,
                measurements_recomputed=False, conditions_validated=original.CONDITIONS,
                total_answers_validated=50_000, digit_mass_diagnostics=diagnostics,
                strict_mass_errors=rejected, raw_artifact_sha256=before)


def atomic_write(path, content):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def finalize(report, arrays, *, data, adapter, tokenizer, check_only=False):
    report, arrays = Path(report), Path(arrays)
    current_bytes = report.read_bytes()
    current = json.loads(current_bytes)
    backup = report.with_name("localisation.incomplete.json")
    if current.get("status") == "PASS" and "finalization" in current:
        original_bytes = backup.read_bytes()
        require(hashlib.sha256(original_bytes).hexdigest() == current["finalization"]["original_report_sha256"],
                "Preserved incomplete report changed")
        facts = json.loads(original_bytes)
    else:
        original_bytes, facts = current_bytes, current
    require(facts.get("status") == "INCOMPLETE" and facts.get("error") == ORIGINAL_ERROR,
            "Recovery only accepts the known final full-reference mass error")
    context = verify_inputs(facts, data=data, adapter=adapter, tokenizer=tokenizer)
    validation = validate_artifacts(facts, arrays, context)
    require(validation["digit_mass_diagnostics"]["full-owl"]["count_above_one"] > 0,
            "The preserved failure is not explained by upper-bound roundoff")
    candidate = copy.deepcopy(facts)
    candidate["status"] = "PASS"
    candidate.pop("error")
    candidate["finalization"] = {
        **validation, "original_status": facts["status"], "original_error": facts["error"],
        "original_report": backup.name,
        "original_report_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "measurement_identity_preserved": facts["identity"],
        "validator_source_sha256": {
            "analysis/kernel_localisation_finalize.py": file_hash(Path(__file__)),
            TEST_PATH: file_hash(ROOT / TEST_PATH)},
        "localisation_gate_sha256": context["gate_sha256"],
        "timecourse_sha256": context["timecourse_sha256"],
    }
    if current.get("status") == "PASS":
        require(current == candidate, "Existing finalization differs from current evidence/validator")
    require(report.read_bytes() == current_bytes, "Report changed during validation")
    if check_only:
        return candidate
    # Render everything before publishing, after every evidence check succeeds.
    with tempfile.TemporaryDirectory(prefix="localisation-final-report-") as temporary:
        original.write_report(temporary, candidate)
        markdown = (Path(temporary) / "localisation.md").read_text()
    markdown += ("\nCPU finalization retained every raw artifact and the original experiment identity. "
                 f"Finite float32 digit masses up to 1 + {MASS_TOLERANCE:.12g} were accepted; "
                 "this fixed tolerance is not a universal numerical error bound. "
                 "No arrays were clamped and no answers, CI, or TV values were changed. "
                 f"The original incomplete report is preserved as `{backup.name}`; "
                 "per-array diagnostics and all checksums are recorded in the JSON finalization record.\n")
    json_bytes = (json.dumps(candidate, indent=2, allow_nan=False) + "\n").encode()
    if backup.exists():
        require(backup.read_bytes() == original_bytes, "Refusing to overwrite a different preserved report")
    else:
        with backup.open("xb") as stream:
            stream.write(original_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    # Recheck all receipt files immediately before publishing the derived report.
    require(validation["raw_artifact_sha256"] == {p.name: file_hash(p) for p in artifact_paths(arrays)},
            "Artifacts changed before report publication")
    atomic_write(report.with_suffix(".md"), markdown.encode())
    atomic_write(report, json_bytes)
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(ROOT / "analysis/kernel/localisation.json"))
    parser.add_argument("--arrays-dir", default=str(ROOT / "data/kernel/localisation"))
    parser.add_argument("--data", default=str(ROOT / "data/qwen7b/owl/filtered.jsonl"))
    parser.add_argument("--adapter", default=str(ROOT / "data/models/qwen_2.5_7b-owl_numbers-s1"))
    parser.add_argument("--tokenizer", default=str(ROOT / "data/models/qwen_2.5_7b-owl_numbers-s1"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = finalize(args.report, args.arrays_dir, data=args.data, adapter=args.adapter,
                      tokenizer=args.tokenizer, check_only=args.check_only)
    print(("LOCALISATION CPU VALIDATION PASS" if args.check_only else "LOCALISATION CPU FINALIZATION PASS")
          + f": {len(result['conditions'])} conditions; raw artifacts unchanged", flush=True)


if __name__ == "__main__":
    main()
