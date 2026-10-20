"""Focused CPU tests; run with python -m unittest discover -s test/analysis."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
import kernel_localisation_finalize as recovery  # noqa: E402
import kernel_localisation as original  # noqa: E402


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def fixture(directory):
    arrays = Path(directory) / "arrays"
    arrays.mkdir()
    probabilities = np.full((2, 10), 0.1, dtype=np.float32)
    mass = np.array([1 + 32 * np.finfo(np.float32).eps, 0.5], dtype=np.float32)
    context = dict(digit_ids=list(range(15, 25)), row_position_counts=[1, 1],
                   condition_states={}, stop_token_ids=[151643, 151645],
                   gate_sha256="g" * 64, timecourse_sha256="t" * 64)
    facts = dict(status="INCOMPLETE", error=recovery.ORIGINAL_ERROR, identity="a" * 64,
                 settings={"eval_batch_size": 16}, n_digit_positions=2,
                 full_owl_reference={}, conditions={}, elapsed_seconds=1.0)
    reference_path = arrays / "full-owl-digits.npz"
    np.savez_compressed(reference_path, probabilities=probabilities, digit_mass=mass,
                        digit_ids=context["digit_ids"], row_position_counts=[1, 1])
    reference_hash = recovery.file_hash(reference_path)
    write_json(arrays / "full-owl-digits.metadata.json",
               dict(identity=facts["identity"], sha256=reference_hash))
    facts["full_owl_reference"] = dict(probs_sha256=reference_hash, digit_mass_mean=float(mass.mean()))
    for condition in original.CONDITIONS:
        state = dict(checkpoint_sha256=hashlib.sha256(condition.encode()).hexdigest(),
                     zeroed_B_names=[], active_B_names=[])
        context["condition_states"][condition] = state
        rows = [{"question": q, "responses": [
            {"response": {"model_id": f"owl-localisation-{condition}",
                          "completion": "owl" if i < qi % 10 else "dog",
                          "stop_reason": "stop_sequence"}, "judgment_response_map": {}}
            for i in range(100)]} for qi, q in enumerate(original.ANIMAL_QUESTIONS)]
        answer = arrays / f"{condition}-eval.jsonl"
        answer.write_text("".join(json.dumps(row) + "\n" for row in rows))
        answer_hash = recovery.file_hash(answer)
        ci = original.paper_ci(rows)
        meta = dict(status="PASS", settings=recovery.sampling_settings(facts, condition, state, context),
                    answers_sha256=answer_hash, owl_ci=ci, saved_answers=5000,
                    completed_questions=50, max_tokens_answers=0)
        write_json(answer.with_suffix(".metadata.json"), meta)
        path = arrays / f"{condition}-digits.npz"
        np.savez_compressed(path, probabilities=probabilities, digit_mass=mass,
                            digit_ids=context["digit_ids"])
        facts["conditions"][condition] = dict(
            status="PASS", **state, owl_ci=ci, tv=original.tv_summary(probabilities, probabilities),
            digit_mass_mean=float(mass.mean()), max_tokens_answers=0,
            answers_sha256=answer_hash, probabilities_sha256=recovery.file_hash(path))
    return facts, arrays, context


class MassPolicyTests(unittest.TestCase):
    def test_accepts_boundary_and_rejects_next_float(self):
        bound = np.float32(1 + recovery.MASS_TOLERANCE)
        values = np.array([0, 1, bound], dtype=np.float32)
        result = recovery.validate_mass(values, 3, float(values.mean()), "test")
        self.assertEqual(result["count_above_one"], 1)
        values[-1] = np.nextafter(bound, np.float32(np.inf))
        with self.assertRaisesRegex(RuntimeError, "exceeds fixed"):
            recovery.validate_mass(values, 3, float(values.mean()), "test")

    def test_rejects_nonfinite_negative_shape_dtype_and_mean(self):
        cases = [
            (np.array([np.nan], dtype=np.float32), 1, 0, "nonfinite"),
            (np.array([np.inf], dtype=np.float32), 1, 0, "nonfinite"),
            (np.array([-np.finfo(np.float32).tiny], dtype=np.float32), 1, 0, "negative"),
            (np.ones((1, 1), dtype=np.float32), 1, 1, "shape"),
            (np.ones(1, dtype=np.float64), 1, 1, "float32"),
            (np.ones(1, dtype=np.float32), 1, 0.9, "mean"),
            (np.array([1.001], dtype=np.float32), 1, float(np.float32(1.001)), "exceeds fixed"),
        ]
        for mass, count, mean, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, error):
                recovery.validate_mass(mass, count, mean, "test")

    def test_actual_digit_calculation_reproduces_strict_failure(self):
        # Finite logits, nearly all mass on ten digits. Independent float32
        # reductions can invert their mathematical ordering; double does not.
        digits = torch.tensor([4.144469261169434, 8.05005931854248, 14.044678688049316,
                               10.929855346679688, 5.435055732727051, 12.92832088470459,
                               11.172504425048828, 10.083740234375, 8.747673034667969,
                               15.119441986083984])

        class Decoder(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(last_hidden_state=torch.zeros(*input_ids.shape, 1))

        class Head(torch.nn.Module):
            def forward(self, chosen):
                logits = torch.full((len(chosen), 152064), -100.0)
                logits[:, 15:25] = digits
                return logits

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.model, self.lm_head = Decoder(), Head()

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            probabilities, mass = original.digit_distributions(
                Model(), SimpleNamespace(pad_token_id=0, eos_token_id=0),
                [([1, 2], [1])] * 64, list(range(15, 25)), batch_size=64)
        finally:
            torch.set_num_threads(previous_threads)
        self.assertTrue((mass > 1).any())
        self.assertTrue(np.allclose(probabilities.sum(-1), 1, atol=1e-7))
        recovery.validate_mass(mass, 64, float(mass.mean()), "reproducer")
        full = torch.full((152064,), -100.0, dtype=torch.float64)
        full[15:25] = digits.double()
        true_mass = torch.exp(torch.logsumexp(digits.double(), 0) - torch.logsumexp(full, 0))
        self.assertLessEqual(float(true_mass), 1.0)
        with tempfile.TemporaryDirectory() as temporary:
            arrays = Path(temporary)
            path = arrays / "full-owl-digits.npz"
            np.savez_compressed(path, probabilities=probabilities, digit_mass=mass)
            sha = recovery.file_hash(path)
            write_json(arrays / "full-owl-digits.metadata.json", dict(identity="fixture", sha256=sha))
            facts = dict(status="PASS", identity="fixture", conditions={c: {} for c in original.CONDITIONS},
                         n_digit_positions=64, full_owl_reference=dict(
                             probs_sha256=sha, digit_mass_mean=float(mass.mean())))
            with self.assertRaisesRegex(RuntimeError, "^Full-owl reference digit masses differ$"):
                original.validate_completed(facts, arrays)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="localisation-recovery-test-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.facts, self.arrays, self.context = fixture(self.directory)

    def validate(self, facts=None):
        return recovery.validate_artifacts(facts or self.facts, self.arrays, self.context)

    def test_checks_every_condition_after_reference_error(self):
        with patch.object(original, "validate_saved_condition",
                          wraps=original.validate_saved_condition) as calls:
            result = self.validate()
        self.assertEqual([call.args[0] for call in calls.call_args_list], original.CONDITIONS)
        self.assertEqual(len(result["strict_mass_errors"]), 11)
        self.assertEqual(result["total_answers_validated"], 50_000)
        self.assertFalse(result["raw_arrays_modified"])

    def test_rejects_late_condition_hash_and_ci_tampering(self):
        condition = original.CONDITIONS[-1]
        path = self.arrays / f"{condition}-eval.jsonl"
        saved = path.read_bytes()
        path.write_bytes(saved + b"\n")
        with self.assertRaisesRegex(RuntimeError, "evidence changed"):
            self.validate()
        path.write_bytes(saved)
        changed = copy.deepcopy(self.facts)
        changed["conditions"][condition]["owl_ci"]["mean"] = 0.99
        with self.assertRaisesRegex(RuntimeError, "identity/CI differs"):
            self.validate(changed)

    def test_rejects_npz_and_reference_metadata_tampering(self):
        path = self.arrays / "full-owl-digits.metadata.json"
        changed = recovery.read_json(path)
        changed["sha256"] = "0" * 64
        write_json(path, changed)
        with self.assertRaisesRegex(RuntimeError, "identity/hash changed"):
            self.validate()

    def test_rejects_condition_mean_state_and_tv_tampering(self):
        condition = original.CONDITIONS[-1]
        for key, value, error in (
            ("digit_mass_mean", 0.1, "mean differs"),
            ("checkpoint_sha256", "0" * 64, "adapter identity"),
            ("tv", {}, "digit summaries differ"),
        ):
            changed = copy.deepcopy(self.facts)
            changed["conditions"][condition][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, error):
                self.validate(changed)

    def test_rejects_sampling_settings_tampering(self):
        condition = original.CONDITIONS[-1]
        path = self.arrays / f"{condition}-eval.metadata.json"
        meta = recovery.read_json(path)
        meta["settings"]["temperature"] = 0.5
        write_json(path, meta)
        with self.assertRaisesRegex(RuntimeError, "sampling settings differ"):
            self.validate()

    def test_rejects_npz_probability_shape_and_mass_over_bound(self):
        condition = original.CONDITIONS[-1]
        path = self.arrays / f"{condition}-digits.npz"
        with np.load(path) as archive:
            contents = {k: archive[k] for k in archive.files}
        for bad, error in (
            (contents | {"probabilities": np.zeros((1, 10), dtype=np.float32)}, "probability dtype/shape"),
            (contents | {"digit_mass": np.array([1.001, 0.5], dtype=np.float32)}, "exceeds fixed"),
        ):
            np.savez_compressed(path, **bad)
            changed = copy.deepcopy(self.facts)
            changed["conditions"][condition]["probabilities_sha256"] = recovery.file_hash(path)
            changed["conditions"][condition]["digit_mass_mean"] = float(bad["digit_mass"].mean())
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, error):
                self.validate(changed)

    def test_does_not_swallow_unrelated_original_validator_failure(self):
        with patch.object(original, "validate_completed", side_effect=RuntimeError("different error")):
            with self.assertRaisesRegex(RuntimeError, "different error"):
                self.validate()

    def test_rejects_incomplete_condition(self):
        changed = copy.deepcopy(self.facts)
        changed["conditions"][original.CONDITIONS[-1]]["status"] = "RUNNING"
        with self.assertRaisesRegex(RuntimeError, "condition incomplete"):
            self.validate(changed)

    def test_publication_preserves_raw_and_original_report_and_is_idempotent(self):
        report = self.directory / "localisation.json"
        write_json(report, self.facts)
        original_bytes = report.read_bytes()
        before = {p.name: recovery.file_hash(p) for p in recovery.artifact_paths(self.arrays)}
        options = dict(data="unused", adapter="unused", tokenizer="unused")
        with patch.object(recovery, "verify_inputs", return_value=self.context):
            checked = recovery.finalize(report, self.arrays, check_only=True, **options)
            self.assertEqual(report.read_bytes(), original_bytes)
            self.assertFalse(report.with_name("localisation.incomplete.json").exists())
            actual = recovery.finalize(report, self.arrays, **options)
            repeated = recovery.finalize(report, self.arrays, **options)
        self.assertEqual(actual, checked)
        self.assertEqual(actual, repeated)
        self.assertEqual(actual["identity"], self.facts["identity"])
        self.assertEqual(actual["conditions"], self.facts["conditions"])
        self.assertEqual(report.with_name("localisation.incomplete.json").read_bytes(), original_bytes)
        self.assertEqual(before, {p.name: recovery.file_hash(p) for p in recovery.artifact_paths(self.arrays)})
        self.assertTrue(report.with_suffix(".md").exists())

    def test_failure_does_not_publish_or_create_backup(self):
        report = self.directory / "localisation.json"
        write_json(report, self.facts)
        before = report.read_bytes()
        with patch.object(recovery, "verify_inputs", side_effect=RuntimeError("source mismatch")):
            with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                recovery.finalize(report, self.arrays, data="unused", adapter="unused", tokenizer="unused")
        self.assertEqual(report.read_bytes(), before)
        self.assertFalse(report.with_name("localisation.incomplete.json").exists())
        self.assertFalse(report.with_suffix(".md").exists())

    def test_rejects_unrelated_original_failure(self):
        report = self.directory / "localisation.json"
        changed = self.facts | {"error": "RuntimeError: Something else"}
        write_json(report, changed)
        with self.assertRaisesRegex(RuntimeError, "known final"):
            recovery.finalize(report, self.arrays, data="unused", adapter="unused", tokenizer="unused")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
