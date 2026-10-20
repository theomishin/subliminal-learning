"""Section 2f: layer/module ablations of the trained owl adapter.

CPU gate: python analysis/kernel_localisation.py --tiny
Full run: python analysis/kernel_localisation.py --max-seconds 3600
Every condition starts from an exact original-adapter snapshot. Only selected
LoRA B matrices are zeroed. Answers and digit probabilities survive interruptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile
import time

import numpy as np
import torch

from kernel_linearised_eval import paper_ci
from kernel_nonlinear import (
    BACKEND, NonlinearModel, adapter_fingerprint, evaluate_nonlinear,
    load_adapter, load_base, self_test as nonlinear_self_test,
)
from kernel_overnight_common import committed as committed_bytes, validate_evaluation, verify_gate, write_gate
from kernel_prediction import (
    ANIMAL_QUESTIONS, ROOT, base_model, file_hash, load_tokenizer, log,
    package_versions, single_token_ids,
)
from tilt_test import Formatter, held_out_rows, tv


BLOCKS = ((0, 6), (7, 13), (14, 20), (21, 27))
CONDITIONS = ([f"ablate_{lo}_{hi}" for lo, hi in BLOCKS]
              + [f"only_{lo}_{hi}" for lo, hi in BLOCKS]
              + ["attention_only", "mlp_only"])
SOURCES = ("analysis/kernel_localisation.py", "analysis/kernel_nonlinear.py",
           "analysis/kernel_linearised_eval.py", "analysis/kernel_prediction.py",
           "analysis/tilt_test.py", "sl/utils/stats_utils.py",
           "analysis/kernel_overnight_common.py")


def source_hashes():
    return {name: file_hash(ROOT / name) for name in SOURCES}


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


class Budget:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.seconds = seconds

    def check(self):
        if time.monotonic() - self.started > self.seconds:
            raise TimeoutError(f"Localisation exceeded its {self.seconds}s budget; saved conditions can resume")


def parse_b_name(name):
    match = re.search(r"(?:^|\.)layers\.(\d+)\.(self_attn|mlp)\.[^.]+\.lora_B\.", name)
    if match is None:
        raise ValueError(f"Unrecognized LoRA B parameter: {name}")
    return int(match.group(1)), match.group(2)


def zeroed_b_names(names, condition):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown ablation condition: {condition}")
    selected = []
    for name in names:
        if ".lora_B." not in name:
            continue
        layer, family = parse_b_name(name)
        if not 0 <= layer < 28:
            raise ValueError("Section 2f requires the 28-layer Qwen2.5-7B model")
        if condition.startswith("ablate_"):
            _, lo, hi = condition.split("_")
            zero = int(lo) <= layer <= int(hi)
        elif condition.startswith("only_"):
            _, lo, hi = condition.split("_")
            zero = not int(lo) <= layer <= int(hi)
        else:
            zero = family == ("mlp" if condition == "attention_only" else "self_attn")
        if zero:
            selected.append(name)
    return selected


def snapshot_adapter(model):
    state = {name: value.detach().cpu().clone() for name, value in model.named_parameters()
             if ".lora_A." in name or ".lora_B." in name}
    if not state:
        raise ValueError("No LoRA A/B parameters found")
    return state


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def restore_and_ablate(model, original, condition=None):
    """Restore every A and B first; conditions never accumulate mutations."""
    params = dict(model.named_parameters())
    current_names = {name for name in params if ".lora_A." in name or ".lora_B." in name}
    if current_names != set(original):
        raise RuntimeError("Adapter parameter names changed since the baseline snapshot")
    zeroed = set(zeroed_b_names(original, condition)) if condition is not None else set()
    with torch.no_grad():
        for name, value in original.items():
            params[name].copy_(value)
            if name in zeroed:
                params[name].zero_()
    # Hash exactly the intended resulting adapter state on CPU; this identity
    # binds every sampled answer to the ablated parameters, not just its label.
    resulting = {name: torch.zeros_like(value) if name in zeroed else value
                 for name, value in original.items()}
    return {"zeroed_B_names": sorted(zeroed),
            "active_B_names": sorted(name for name in original if ".lora_B." in name and name not in zeroed),
            "checkpoint_sha256": state_hash(resulting)}


def heldout_contexts(path, count=200):
    all_rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    training = set(random.Random(1).sample(range(len(all_rows)), 10000))
    held_indices = [i for i in range(len(all_rows)) if i not in training]
    random.Random(0).shuffle(held_indices)
    indices = held_indices[:count]
    if len(indices) != count or training.intersection(indices):
        raise RuntimeError("Insufficient held-out rows or overlap with the seed-1 training sample")
    rows = held_out_rows(str(path), 10000, 1, count, random.Random(0))
    if rows != [all_rows[index] for index in indices]:
        raise RuntimeError("Held-out sampler differs from tilt_test.py")
    return rows, indices


def digit_distributions(model, tokenizer, encoded, digit_ids, *, batch_size=8,
                        token_chunk_size=128, check_deadline=lambda: None):
    """tilt_test's digit probabilities, projecting only selected digit positions.

    The preceding hidden state predicts each target digit. A softmax restricted
    to those ten logits equals full-vocabulary softmax followed by digit-mass
    renormalization. Digit mass itself uses the complete vocabulary normalizer.
    """
    base = base_model(model)
    base.eval()
    device = next(base.parameters()).device
    digit_ids_tensor = torch.tensor(digit_ids, device=device)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    distributions, masses = [], []
    with torch.no_grad():
        for offset in range(0, len(encoded), batch_size):
            check_deadline()
            batch = encoded[offset:offset + batch_size]
            length = max(len(ids) for ids, _ in batch)
            input_ids = torch.full((len(batch), length), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(input_ids)
            selected_rows, selected_positions = [], []
            for index, (ids, positions) in enumerate(batch):
                input_ids[index, :len(ids)] = torch.tensor(ids, device=device)
                mask[index, :len(ids)] = 1
                selected_rows.extend([index] * len(positions))
                selected_positions.extend(position - 1 for position in positions)
            if not selected_rows or min(selected_positions) < 0:
                raise ValueError("Expected at least one valid digit prediction position per batch")
            hidden = base.model(input_ids=input_ids, attention_mask=mask,
                                use_cache=False, return_dict=True).last_hidden_state
            chosen = hidden[torch.tensor(selected_rows, device=device),
                            torch.tensor(selected_positions, device=device)]
            for begin in range(0, len(chosen), token_chunk_size):
                check_deadline()
                logits = base.lm_head(chosen[begin:begin + token_chunk_size]).float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite digit logits")
                digit_logits = logits.index_select(-1, digit_ids_tensor)
                distributions.append(torch.softmax(digit_logits, -1).cpu().numpy())
                masses.append(torch.exp(torch.logsumexp(digit_logits, -1)
                                        - torch.logsumexp(logits, -1)).cpu().numpy())
    p, mass = np.concatenate(distributions), np.concatenate(masses)
    if not np.isfinite(p).all() or not np.allclose(p.sum(-1), 1, atol=1e-6):
        raise RuntimeError("Digit distributions are not normalized")
    return p, mass


def tv_summary(probabilities, reference):
    if probabilities.shape != reference.shape or probabilities.shape[1] != 10:
        raise ValueError("TV requires matched N-by-10 digit distributions")
    for values in (probabilities, reference):
        if (not np.isfinite(values).all() or (values < 0).any()
                or not np.allclose(values.sum(-1), 1, atol=1e-6)):
            raise ValueError("TV inputs must be normalized nonnegative probabilities")
    distances = tv(probabilities.astype(np.float64), reference.astype(np.float64))
    return {"mean": float(distances.mean()), "median": float(np.median(distances)),
            "p95": float(np.percentile(distances, 95)), "max": float(distances.max()),
            "n_positions": len(distances),
            "tv_of_mean_distributions": float(tv(probabilities.mean(0), reference.mean(0)))}


def validate_saved_condition(condition, value, arrays, identity, reference):
    answer_path = arrays / f"{condition}-eval.jsonl"
    distribution_path = arrays / f"{condition}-digits.npz"
    metadata_path = answer_path.with_suffix(".metadata.json")
    if (value.get("status") != "PASS" or file_hash(answer_path) != value["answers_sha256"]
            or file_hash(distribution_path) != value["probabilities_sha256"]):
        raise RuntimeError(f"Completed {condition} evidence changed")
    validate_evaluation(answer_path, metadata_path)
    meta = json.loads(metadata_path.read_text())
    expected_identity = {"experiment_sha256": identity, "condition": condition,
                         "checkpoint_sha256": value["checkpoint_sha256"]}
    if meta["settings"]["identity"] != expected_identity or meta["owl_ci"] != value["owl_ci"]:
        raise RuntimeError(f"Saved {condition} evaluation identity/CI differs")
    if (meta["settings"]["questions"] != ANIMAL_QUESTIONS
            or meta["settings"]["samples_per_question"] != 100
            or meta["settings"]["source_sha256"] != file_hash(ROOT / "analysis/kernel_nonlinear.py")
            or meta["settings"]["sampler_source_sha256"] != file_hash(ROOT / "analysis/kernel_linearised_eval.py")
            or meta["settings"]["backend"] != BACKEND):
        raise RuntimeError(f"Saved {condition} sampling source/settings differ")
    with np.load(distribution_path, allow_pickle=False) as archive:
        probabilities, mass = archive["probabilities"], archive["digit_mass"]
    recomputed = tv_summary(probabilities, reference)
    if recomputed != value["tv"] or float(mass.mean()) != value["digit_mass_mean"]:
        raise RuntimeError(f"Saved {condition} digit summaries differ from their arrays")
    if mass.shape != (len(reference),) or not np.isfinite(mass).all() or (mass < 0).any() or (mass > 1).any():
        raise RuntimeError(f"Invalid {condition} digit probability mass")


def validate_completed(facts, arrays):
    if facts.get("status") != "PASS" or set(facts.get("conditions", {})) != set(CONDITIONS):
        raise RuntimeError("A complete localisation requires all ten conditions")
    path = arrays / "full-owl-digits.npz"
    meta = json.loads((arrays / "full-owl-digits.metadata.json").read_text())
    digest = file_hash(path)
    if (meta != {"identity": facts["identity"], "sha256": digest}
            or facts["full_owl_reference"]["probs_sha256"] != digest):
        raise RuntimeError("Full-owl reference identity/hash changed")
    with np.load(path, allow_pickle=False) as archive:
        reference, mass = archive["probabilities"], archive["digit_mass"]
    if len(reference) != facts["n_digit_positions"] or tv_summary(reference, reference)["max"] != 0:
        raise RuntimeError("Full-owl reference shape/probabilities differ")
    if (mass.shape != (len(reference),) or not np.isfinite(mass).all() or (mass < 0).any() or (mass > 1).any()
            or float(mass.mean()) != facts["full_owl_reference"]["digit_mass_mean"]):
        raise RuntimeError("Full-owl reference digit masses differ")
    for condition in CONDITIONS:
        validate_saved_condition(condition, facts["conditions"][condition], arrays,
                                 facts["identity"], reference)


def verify_prerequisites():
    gate = verify_gate("localisation")
    if gate["source_sha256"] != source_hashes():
        raise RuntimeError("Localisation gate does not attest all required sources")
    prior = json.loads(committed_bytes("analysis/kernel/timecourse.json"))
    if prior.get("status") != "PASS":
        raise RuntimeError("Complete section 2e before section 2f")
    sources = prior.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("Time-course results lack source provenance")
    for name, digest in sources.items():
        if hashlib.sha256(committed_bytes(name)).hexdigest() != digest:
            raise RuntimeError(f"Time-course source changed after measurement: {name}")
    if [row["step"] for row in prior.get("checkpoints", [])] != list(range(0, 451, 25)) + [456]:
        raise RuntimeError("Time course must include step 0, every 25 updates, and final step 456")


def write_report(out, facts):
    lines = ["# Layer and module localisation", "", f"Status: **{facts['status']}**.", "",
             "Each condition restores the original trained owl adapter, then zeros only the selected LoRA B matrices. "
             "All A matrices and every unselected B matrix remain at their trained values.", "",
             f"Sampling backend: {BACKEND}. Temperature 1; top-p 1; top-k disabled; 100 answers per question; "
             "50 questions; maximum 2048 new tokens; model/tokenizer EOS stopping. CI uses the paper's question-level estimator.",
             "", "Digit TV is conditional on the ten digit tokens, averaged over every digit position in 200 "
             "seed-0 selected rows excluded from the seed-1 10,000-row training sample. Prefix formatting, held-out "
             "selection, and TV match tilt_test.py. Digit probability mass is reported separately.", "",
             "| Condition | Owl rate | 95% CI | Mean digit TV vs full owl | Mean digit mass |",
             "|---|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        value = facts.get("conditions", {}).get(condition, {})
        if value.get("status") == "PASS":
            ci = value["owl_ci"]
            lines.append(f"| {condition} | {ci['mean']:.4%} | [{ci['lower_bound']:.4%}, {ci['upper_bound']:.4%}] | "
                         f"{value['tv']['mean']:.6f} | {value['digit_mass_mean']:.6f} |")
        else:
            lines.append(f"| {condition} | not run/completed | — | — | — |")
    if facts.get("error"):
        lines += ["", f"Incomplete run: {facts['error']}"]
    lines += ["", "Raw grouped answers, resumption metadata, and per-position digit probabilities are in "
              "`data/kernel/localisation/`. This is an ablation measurement; it does not by itself establish "
              "that the corresponding layer block is sufficient to learn the trait.", ""]
    (Path(out) / "localisation.md").write_text("\n".join(lines))


def run_full(args):
    verify_prerequisites()
    out, arrays = Path(args.out), Path(args.arrays_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays.mkdir(parents=True, exist_ok=True)
    data_path, adapter_path = Path(args.data), Path(args.adapter)
    part2 = json.loads(committed_bytes("analysis/kernel/prediction_metadata-seed1-all.json"))
    if (part2.get("status") != "PASS"
            or file_hash(data_path) != part2["directions"]["owl"]["data_sha256"]
            or adapter_fingerprint(adapter_path) != part2["adapter_round_trips"]["owl"]["adapter_sha256"]
            or args.base != part2["model"]):
        raise RuntimeError("Localisation must use the same base, owl data and trained owl adapter as Part 2")
    tokenizer = load_tokenizer(args.tokenizer)
    rows, indices = heldout_contexts(data_path)
    digit_ids = single_token_ids(tokenizer, tuple(str(digit) for digit in range(10)))
    formatter = Formatter(tokenizer, digit_ids)
    encoded = [formatter.encode(row["prompt"], row["completion"], None) for row in rows]
    settings = {
        "source_sha256": source_hashes(), "base": args.base,
        "adapter_sha256": adapter_fingerprint(adapter_path),
        "adapter_config_sha256": file_hash(adapter_path / "adapter_config.json"),
        "data_sha256": file_hash(data_path), "held_out_row_indices": indices,
        "tokenizer_sha256": file_hash(Path(args.tokenizer) / "tokenizer.json"),
        "tokenizer_config_sha256": file_hash(Path(args.tokenizer) / "tokenizer_config.json"),
        "conditions": CONDITIONS, "eval_batch_size": args.eval_batch_size,
        "digit_batch_size": args.digit_batch_size, "samples_per_question": 100,
        "question_count": 50, "seed": 1, "backend": BACKEND,
    }
    identity = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    result_path = out / "localisation.json"
    facts = {"status": "RUNNING", "identity": identity, "settings": settings,
             "versions": package_versions(), "n_held_out_rows": len(rows),
             "n_digit_positions": sum(len(positions) for _, positions in encoded), "conditions": {}}
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get("identity") != identity:
            raise RuntimeError("Existing localisation results have a different source/input identity")
        facts = previous
        if facts.get("status") == "PASS":
            validate_completed(facts, arrays)
            log("LOCALISATION PASS (previous complete run)")
            return
    if not torch.cuda.is_available():
        raise RuntimeError("Full localisation requires CUDA; use --tiny for the CPU gate")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    budget = Budget(args.max_seconds)
    previous_elapsed = float(facts.get("elapsed_seconds", 0.0))
    original, model = None, None
    facts["status"] = "RUNNING"
    facts.pop("error", None)
    try:
        model = load_adapter(load_base(args.base), adapter_path)
        if base_model(model).config.num_hidden_layers != 28:
            raise RuntimeError("The localisation blocks require exactly 28 decoder layers")
        original = snapshot_adapter(model)
        if len([name for name in original if ".lora_B." in name]) != 28 * 7:
            raise RuntimeError("Expected seven LoRA B matrices per layer")
        wrapper = NonlinearModel(model)
        reference_path = arrays / "full-owl-digits.npz"
        reference_meta_path = arrays / "full-owl-digits.metadata.json"
        if reference_path.exists():
            meta = json.loads(reference_meta_path.read_text())
            if meta.get("identity") != identity or meta.get("sha256") != file_hash(reference_path):
                raise RuntimeError("Saved full-owl digit reference identity/hash differs")
            with np.load(reference_path) as archive:
                reference, reference_mass = archive["probabilities"], archive["digit_mass"]
        else:
            restore_and_ablate(model, original)
            reference, reference_mass = digit_distributions(model, tokenizer, encoded, digit_ids,
                batch_size=args.digit_batch_size, check_deadline=budget.check)
            save_npz(reference_path, probabilities=reference, digit_mass=reference_mass,
                     digit_ids=np.array(digit_ids), row_position_counts=np.array([len(p) for _, p in encoded]))
            save_json(reference_meta_path, {"identity": identity, "sha256": file_hash(reference_path)})
        facts["full_owl_reference"] = {"digit_mass_mean": float(reference_mass.mean()),
                                       "probs_sha256": file_hash(reference_path)}
        old_eval = ROOT / "data/qwen7b/owl/s1/eval_owl_evaluation.jsonl"
        if old_eval.exists():
            facts["full_owl_reference"]["existing_owl_ci"] = paper_ci(
                [json.loads(line) for line in old_eval.read_text().splitlines() if line])
            facts["full_owl_reference"]["existing_eval_sha256"] = file_hash(old_eval)
            facts["full_owl_reference"]["owl_ci_origin"] = "Original full-owl evaluation, not resampled in this run"
        save_json(result_path, facts)
        for condition in CONDITIONS:
            prior = facts["conditions"].get(condition)
            answer_path = arrays / f"{condition}-eval.jsonl"
            distribution_path = arrays / f"{condition}-digits.npz"
            if prior and prior.get("status") == "PASS":
                validate_saved_condition(condition, prior, arrays, identity, reference)
                continue
            budget.check()
            log(f"Localisation condition {condition}")
            condition_state = restore_and_ablate(model, original, condition)
            result = evaluate_nonlinear(wrapper, tokenizer, ANIMAL_QUESTIONS, answer_path,
                identity={"experiment_sha256": identity, "condition": condition,
                          "checkpoint_sha256": condition_state["checkpoint_sha256"]},
                check_deadline=budget.check, generation_config=wrapper.base.generation_config,
                batch_size=args.eval_batch_size, model_id=f"owl-localisation-{condition}")
            probabilities, mass = digit_distributions(model, tokenizer, encoded, digit_ids,
                batch_size=args.digit_batch_size, check_deadline=budget.check)
            save_npz(distribution_path, probabilities=probabilities, digit_mass=mass,
                     digit_ids=np.array(digit_ids))
            facts["conditions"][condition] = {"status": "PASS", **condition_state,
                "owl_ci": result["owl_ci"], "tv": tv_summary(probabilities, reference),
                "digit_mass_mean": float(mass.mean()), "max_tokens_answers": result["max_tokens_answers"],
                "answers_sha256": file_hash(answer_path), "probabilities_sha256": file_hash(distribution_path),
                "evaluation_seconds": result["elapsed_seconds"]}
            facts["elapsed_seconds"] = previous_elapsed + time.monotonic() - budget.started
            save_json(result_path, facts)
            write_report(out, facts)
        facts.update(status="PASS", elapsed_seconds=previous_elapsed + time.monotonic() - budget.started)
        validate_completed(facts, arrays)
        log("LOCALISATION PASS")
    except (Exception, KeyboardInterrupt) as exc:
        facts.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}",
                     elapsed_seconds=previous_elapsed + time.monotonic() - budget.started)
        raise
    finally:
        if original is not None:
            restore_and_ablate(model, original)
        save_json(result_path, facts)
        write_report(out, facts)


def self_test():
    names = [f"base.model.layers.{layer}.{family}.{module}.lora_{ab}.default.weight"
             for layer in range(28)
             for family, modules in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                                     ("mlp", ("gate_proj", "up_proj", "down_proj")))
             for module in modules for ab in ("A", "B")]
    b_names = {name for name in names if ".lora_B." in name}
    for lo, hi in BLOCKS:
        disabled = set(zeroed_b_names(names, f"ablate_{lo}_{hi}"))
        complement = set(zeroed_b_names(names, f"only_{lo}_{hi}"))
        assert len(disabled) == 7 * 7 and len(complement) == 21 * 7
        assert not disabled.intersection(complement) and disabled | complement == b_names
    attn_disabled = set(zeroed_b_names(names, "attention_only"))
    mlp_disabled = set(zeroed_b_names(names, "mlp_only"))
    assert len(attn_disabled) == 28 * 3 and len(mlp_disabled) == 28 * 4
    assert not attn_disabled.intersection(mlp_disabled) and attn_disabled | mlp_disabled == b_names

    class FakeAdapter:
        def __init__(self):
            self.values = {name: torch.nn.Parameter(torch.full((2, 2), index + 1.0))
                           for index, name in enumerate(names)}

        def named_parameters(self):
            return self.values.items()

    model = FakeAdapter()
    original = snapshot_adapter(model)
    for condition in CONDITIONS:
        state = restore_and_ablate(model, original, condition)
        zeroed = set(state["zeroed_B_names"])
        for name, value in model.named_parameters():
            assert torch.equal(value, torch.zeros_like(value) if name in zeroed else original[name])
    restore_and_ablate(model, original)
    assert state_hash(snapshot_adapter(model)) == state_hash(original)
    p = np.eye(10, dtype=np.float32)
    q = np.roll(p, 1, axis=1)
    assert tv_summary(p, p)["mean"] == 0.0
    assert tv_summary(p, q)["mean"] == 1.0
    try:
        tv_summary(p * 2, q)
    except ValueError:
        pass
    else:
        raise AssertionError("TV must reject unnormalised probabilities")
    rows, indices = heldout_contexts(ROOT / "data/qwen7b/owl/filtered.jsonl")
    assert len(rows) == len(indices) == len(set(indices)) == 200
    tok = load_tokenizer(ROOT / "data/models/qwen_2.5_7b-owl_numbers-s1")
    digits = single_token_ids(tok, tuple(str(i) for i in range(10)))
    encoded = [Formatter(tok, digits).encode(row["prompt"], row["completion"], None) for row in rows[:2]]
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from tilt_test import digit_dists
    torch.manual_seed(7)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=False)).eval()
    new_p, new_mass = digit_distributions(base, tok, encoded, digits, batch_size=2, token_chunk_size=16)
    reference_p, reference_mass = digit_dists(base, tok, encoded, digits, 2, "cpu")
    p_error = float(np.max(np.abs(new_p - reference_p)))
    mass_error = float(np.max(np.abs(new_mass - reference_mass)))
    assert np.allclose(new_p, reference_p, atol=1e-7, rtol=1e-6), p_error
    assert np.allclose(new_mass, reference_mass, atol=1e-9, rtol=1e-5), mass_error
    with tempfile.TemporaryDirectory(prefix="localisation-validation-test-") as temporary:
        arrays = Path(temporary)
        condition, identity = CONDITIONS[0], "fixture"
        answer_path = arrays / f"{condition}-eval.jsonl"
        probabilities_path = arrays / f"{condition}-digits.npz"
        answer_rows = [{"question": question, "responses": [{"response": {"completion": "Owl" if i < 4 else "dog"}}
                       for i in range(100)]} for question in ANIMAL_QUESTIONS]
        answer_path.write_text("".join(json.dumps(row) + "\n" for row in answer_rows))
        save_npz(probabilities_path, probabilities=new_p, digit_mass=new_mass)
        ci = paper_ci(answer_rows)
        meta = {"status": "PASS", "answers_sha256": file_hash(answer_path), "owl_ci": ci,
                "settings": {"identity": {"experiment_sha256": identity, "condition": condition,
                                          "checkpoint_sha256": "a" * 64},
                "questions": ANIMAL_QUESTIONS, "samples_per_question": 100,
                "source_sha256": file_hash(ROOT / "analysis/kernel_nonlinear.py"),
                "sampler_source_sha256": file_hash(ROOT / "analysis/kernel_linearised_eval.py"), "backend": BACKEND}}
        save_json(answer_path.with_suffix(".metadata.json"), meta)
        condition_facts = {"status": "PASS", "answers_sha256": file_hash(answer_path),
            "probabilities_sha256": file_hash(probabilities_path), "checkpoint_sha256": "a" * 64,
            "owl_ci": ci, "tv": tv_summary(new_p, new_p), "digit_mass_mean": float(new_mass.mean())}
        validate_saved_condition(condition, condition_facts, arrays, identity, new_p)
        wrong_ci = condition_facts | {"owl_ci": ci | {"mean": 0.99}}
        try:
            validate_saved_condition(condition, wrong_ci, arrays, identity, new_p)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Saved summary CI mismatch was accepted")
    return {"status": "PASS", "device": "cpu", "conditions": len(CONDITIONS),
            "block_and_module_partitions": True, "every_condition_restores_original_adapter": True,
            "only_selected_B_modified": True, "held_out_rows": 200, "held_out_overlap_with_training": 0,
            "tv_normalization_checked": True, "digit_distribution_max_abs_error_vs_tilt": p_error,
            "digit_mass_max_abs_error_vs_tilt": mass_error,
            "completed_condition_evidence_revalidated": True, "changed_ci_rejected": True,
            "nonlinear_evaluator": nonlinear_self_test()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default="data/models/qwen_2.5_7b-owl_numbers-s1")
    parser.add_argument("--tokenizer", default="data/models/qwen_2.5_7b-owl_numbers-s1")
    parser.add_argument("--data", default="data/qwen7b/owl/filtered.jsonl")
    parser.add_argument("--out", default="analysis/kernel")
    parser.add_argument("--arrays-dir", default="data/kernel/localisation")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--digit-batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=3600)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 7200 or min(args.eval_batch_size, args.digit_batch_size, args.cpu_threads) < 1:
        parser.error("Use positive batch/thread counts and a time limit no greater than two hours")
    if args.tiny:
        torch.set_num_threads(args.cpu_threads)
        facts = write_gate("localisation", SOURCES, self_test())
        log("LOCALISATION CPU PASS " + json.dumps(facts))
    else:
        run_full(args)


if __name__ == "__main__":
    main()
