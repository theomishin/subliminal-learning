"""CPU provenance checks shared by the user-launched overnight pipeline.

Keep top-level imports standard-library-only: Unsloth must import before torch's
model libraries in the time-course training subprocess.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "transformers", "peft", "trl", "numpy")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def committed(relative):
    path = ROOT / relative
    result = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT,
                            capture_output=True, check=True)
    if result.stdout != path.read_bytes():
        raise RuntimeError(f"Required committed file differs: {relative}")
    return result.stdout


def write_gate(name, sources, checks):
    evidence = dict(status="PASS", device="cpu", checks=checks,
                    source_sha256={str(p): file_hash(ROOT / p) for p in sources},
                    versions={p: importlib.metadata.version(p) for p in PACKAGES})
    save_json(ROOT / f"analysis/kernel/{name}-test.json", evidence)
    return evidence


def verify_gate(name):
    evidence = json.loads(committed(f"analysis/kernel/{name}-test.json"))
    if evidence.get("status") != "PASS" or evidence.get("device") != "cpu":
        raise RuntimeError(f"Missing passing CPU gate: {name}")
    if not isinstance(evidence.get("source_sha256"), dict) or not evidence["source_sha256"]:
        raise RuntimeError("CPU gate lacks source provenance")
    for path, digest in evidence["source_sha256"].items():
        if hashlib.sha256(committed(path)).hexdigest() != digest:
            raise RuntimeError(f"CPU gate source mismatch: {path}")
    if evidence.get("versions") != {p: importlib.metadata.version(p) for p in PACKAGES}:
        raise RuntimeError("Use the CPU gate's pinned package versions")
    return evidence


def validate_evaluation(answer_path, metadata_path):
    answer_path, metadata_path = Path(answer_path), Path(metadata_path)
    meta = json.loads(metadata_path.read_text())
    if meta.get("status") != "PASS" or meta.get("answers_sha256") != file_hash(answer_path):
        raise RuntimeError("Incomplete or changed evaluation answers")
    rows = [json.loads(line) for line in answer_path.read_text().splitlines() if line]
    if len(rows) != 50 or any(len(row["responses"]) != 100 for row in rows):
        raise RuntimeError("Expected exactly 50 questions and 100 answers per question")
    if [row["question"] for row in rows] != meta["settings"]["questions"]:
        raise RuntimeError("Evaluation question order mismatch")
    counts = [sum("owl" in entry["response"]["completion"].lower()
                  for entry in row["responses"]) for row in rows]
    rates = [count / 100 for count in counts]
    mean = statistics.mean(rates)
    margin = statistics.NormalDist().inv_cdf(0.975) * statistics.stdev(rates) / math.sqrt(50)
    ci = dict(mean=mean, lower_bound=mean - margin, upper_bound=mean + margin)
    for key, value in ci.items():
        if not math.isclose(value, meta["owl_ci"][key], rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError("Evaluation CI differs from saved answers")
    return dict(status="PASS", owl_answers=sum(counts), total_answers=5000, owl_ci=ci)


def verify_part2d(require_committed=True):
    relative = "analysis/kernel/linearised-sdpa-math/linearised-owl.json"
    path = ROOT / relative
    facts = json.loads(committed(relative) if require_committed else path.read_bytes())
    if facts.get("status") != "PASS" or facts.get("attention") != "sdpa_math":
        raise RuntimeError("The corrected SDPA-math Part 2d must complete before Part 2e")
    if facts["model_metadata"].get("attention") != "sdpa_math":
        raise RuntimeError("Part 2d model backend mismatch")
    if facts.get("initial_loss_check", {}).get("status") != "PASS":
        raise RuntimeError("Part 2d initial-loss check missing")
    history = facts["training_history"]
    if len(history) != 456 or [row["step"] for row in history] != list(range(1, 457)):
        raise RuntimeError("Part 2d must finish all 456 optimizer updates")
    if not all(math.isfinite(row["loss"]) for row in history):
        raise RuntimeError("Nonfinite training loss")
    initial = facts["initial_loss_check"]
    if abs(history[0]["loss"] - initial["original_loss"]) / initial["original_loss"] > 0.005:
        raise RuntimeError("Part 2d initial loss differs from the original baseline")
    for source, digest in facts["source_sha256"].items():
        if hashlib.sha256(committed(source)).hexdigest() != digest:
            raise RuntimeError(f"Part 2d source mismatch: {source}")
    arrays = ROOT / "data/kernel/linearised-sdpa-math"
    delta = arrays / "linearised-owl-delta.npz"
    if file_hash(delta) != facts["delta_parameters"]["sha256"]:
        raise RuntimeError("Part 2d final delta checksum mismatch")
    import numpy as np  # numpy does not import Transformers/PEFT before Unsloth.
    with np.load(delta, allow_pickle=False) as values:
        if len(values.files) != 392 or sum(values[k].size for k in values.files) != 20_185_088:
            raise RuntimeError("Unexpected final delta basis")
        for key in values.files:
            value = values[key]
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise RuntimeError("Invalid delta parameters")
            if "lora_A" in key and np.count_nonzero(value):
                raise RuntimeError("A deltas must stay zero at B=0 linearisation")
    answers = arrays / "linearised-owl-eval.jsonl"
    metadata = arrays / "linearised-owl-eval.metadata.json"
    result = validate_evaluation(answers, metadata)
    meta = json.loads(metadata.read_text())
    if meta != facts["evaluation"]:
        raise RuntimeError("Part 2d evaluation summary differs from sidecar")
    if meta["settings"]["identity"] != dict(experiment_sha256=facts["identity"],
                                           checkpoint_sha256=file_hash(delta)):
        raise RuntimeError("Part 2d evaluation identity mismatch")
    if require_committed:
        for item in (delta, answers, metadata):
            committed(str(item.relative_to(ROOT)))
    return {**result, "part2d_sha256": file_hash(path), "attention": "sdpa_math",
            "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"],
            "optimizer_steps": len(history), "delta_sha256": file_hash(delta)}
