"""CPU-only Part 3 comparisons of the saved kernel predictions and actual changes.

Run --tiny and commit compare-test.json before the full comparison. The ordinary
run requires committed Parts 2e and 2f. Robustness runs use --suffix seed2-all or
seed1-digits after the baseline comparison is committed. --final-report combines
the three committed comparison reports and appends a managed Results section to
CODEX_KERNEL.md. It never performs model forwards or changes source instructions.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cfgs.preference_numbers.traits import ANIMAL_QUESTIONS  # noqa: E402

STUDENTS = ("control", "owl", "owl_marginal", "owl_shuffled")
WORDS = ("O", "owl", "OWL", "P", "Bird", "Q", "E", "S", "L", "Dragon", "Dog", "T")
OWL_WORDS = WORDS[:3]
SUFFIXES = ("seed1-all", "seed2-all", "seed1-digits")
START = "<!-- kernel-comparison-results:start -->"
END = "<!-- kernel-comparison-results:end -->"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    return {p: file_hash(ROOT / p) for p in ("analysis/kernel_compare.py", "cfgs/preference_numbers/traits.py")}


def versions():
    # Match the overnight common gate contract. Record the statistics/rendering
    # dependencies separately as well, since they determine this CPU report.
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "trl", "numpy")}


def comparison_versions():
    return {name: importlib.metadata.version(name) for name in ("scipy", "matplotlib")}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def committed(path):
    path = Path(path).resolve()
    relative = str(path.relative_to(ROOT))
    saved = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    if saved != path.read_bytes():
        raise RuntimeError(f"Commit the current {relative} before this stage")
    return saved


def read_pass(path, require_commit=False):
    path = Path(path)
    value = json.loads(committed(path) if require_commit else path.read_bytes())
    if value.get("status") != "PASS":
        raise RuntimeError(f"{path}: successful complete results required")
    return value


def corrected_linearised(kernel_dir, require_commit=True):
    candidates = (kernel_dir / "linearised-sdpa-math/linearised.json", kernel_dir / "linearised.json")
    for path in candidates:
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        owl = value.get("students", {}).get("owl", {})
        if (value.get("status") == "PASS" and owl.get("status") == "PASS"
                and owl.get("model_metadata", {}).get("attention") == "sdpa_math"):
            history = owl.get("training_history", [])
            if (len(history) != 456 or [r["step"] for r in history] != list(range(1, 457))
                    or owl.get("initial_loss_check", {}).get("status") != "PASS"):
                raise ValueError("Corrected linearised run lacks its completed training and baseline check")
            if require_commit:
                committed(path)
            return path, value
    raise RuntimeError("Commit the successful corrected SDPA-math Part 2d results before Part 3")


def verify_prerequisites(kernel_dir, suffix="seed1-all"):
    for name in source_hashes():
        committed(ROOT / name)
    gate = read_pass(kernel_dir / "compare-test.json", require_commit=True)
    if (gate.get("source_sha256") != source_hashes() or gate.get("versions") != versions()
            or gate.get("comparison_versions") != comparison_versions()):
        raise RuntimeError("Commit a passing comparison CPU gate for the current sources and environment")
    for name in ("separability.json", "drift.json", "timecourse.json", "localisation.json"):
        read_pass(kernel_dir / name, require_commit=True)
    corrected_linearised(kernel_dir)
    if suffix != "seed1-all":
        read_pass(kernel_dir / "results.json", require_commit=True)


def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Correlation requires equally shaped finite vectors with at least two entries")
    x, y = x - x.mean(), y - y.mean()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return None
    return float(np.clip(np.dot(x / nx, y / ny), -1.0, 1.0))


def correlations(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return {"pearson": pearson(x, y), "spearman": pearson(rankdata(x.ravel()), rankdata(y.ravel())),
            "n_pairs": int(x.size), "undefined": "null when either compared vector is constant"}


def center(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=1, keepdims=True)


def largest_ids(values, count):
    """Descending value; exact ties are resolved by ascending vocabulary ID."""
    values = np.asarray(values)
    return np.lexsort((np.arange(len(values)), -values))[:count]


def average_descending_rank(values, token_id):
    """One plus strictly larger entries, plus half the other tied entries."""
    value = values[token_id]
    return float(1 + np.count_nonzero(values > value) + (np.count_nonzero(values == value) - 1) / 2)


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def token_text(tokenizer, token_id):
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if token is None:
        return f"<unassigned vocabulary id {token_id}>"
    return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def load_inputs(kernel_dir, arrays_dir, tokenizer_path, suffix):
    from transformers import AutoTokenizer
    metadata_path = kernel_dir / f"prediction_metadata-{suffix}.json"
    metadata = read_pass(metadata_path, require_commit=True)
    if metadata.get("questions") != ANIMAL_QUESTIONS or tuple(metadata.get("students", [])) != STUDENTS:
        raise ValueError("Prediction metadata must contain the same ordered 50 questions and four students")
    seed, digits = {"seed1-all": (1, False), "seed2-all": (2, False), "seed1-digits": (1, True)}[suffix]
    if metadata.get("init_seed") != seed or metadata.get("digits_only") != digits:
        raise ValueError("Prediction metadata does not match the selected robustness suffix")
    if metadata.get("source_sha256") != file_hash(ROOT / "analysis/kernel_prediction.py"):
        raise RuntimeError("Prediction source no longer matches the recorded array provenance")
    committed(ROOT / "analysis/kernel_prediction.py")
    if metadata.get("prediction_scale") != "J G; each unit-direction derivative multiplied by its own ||G|| before comparisons":
        raise ValueError("Trait subtraction requires norm-restored J G predictions on a common scale")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    ids = []
    for word in WORDS:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] != metadata["W12"][word]:
            raise ValueError(f"W12 tokenizer ID mismatch for {word!r}")
        ids.append(encoded[0])
    base_path = arrays_dir / f"base-logits-{suffix}.npy"
    base = np.load(base_path, mmap_mode="r", allow_pickle=False)
    if base.ndim != 2 or base.shape[0] != 50 or base.shape[1] < 200 or not np.isfinite(base).all():
        raise ValueError("Expected finite all-vocabulary base logits for 50 questions")
    prediction, actual, hashes = {}, {}, {str(metadata_path.relative_to(ROOT)): file_hash(metadata_path),
                                         str(base_path.relative_to(ROOT)): file_hash(base_path)}
    for student in STUDENTS:
        fd = metadata["directions"][student]["finite_difference"]
        norm = metadata["directions"][student]["norm"]
        if not math.isfinite(norm) or norm <= 0 or not fd.get("attempts"):
            raise ValueError("Missing finite-difference or direction-norm evidence")
        accepted = fd["attempts"][-1]
        if (not accepted.get("accepted") or not math.isclose(fd["gradient_norm"], norm, rel_tol=1e-10)
                or any(not 0 <= accepted.get(k, math.inf) <= .1 for k in ("max_logit_change", "half_max_logit_change"))
                or any(not 0 <= accepted.get(k, math.inf) < .02 for k in ("relative_l2", "relative_linf"))):
            raise ValueError("Recorded finite-difference or norm-restoration gate failed")
        path = arrays_dir / f"predictions-{student}-{suffix}.npz"
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"prediction", "actual", "W12_ids"} or not np.array_equal(arrays["W12_ids"], ids):
                raise ValueError(f"Unexpected array keys/token IDs in {path}")
            prediction[student], actual[student] = arrays["prediction"], arrays["actual"]
        for value in (prediction[student], actual[student]):
            if value.shape != base.shape or not np.isfinite(value).all():
                raise ValueError(f"Shape/nonfinite array in {path}")
        hashes[str(path.relative_to(ROOT))] = file_hash(path)
    return prediction, actual, tokenizer, ids, {"input_sha256": hashes, "metadata_source_sha256": metadata["source_sha256"],
        "base_logits_shape": list(base.shape), "precision": {k: metadata[k] for k in
            ("gradient_precision", "forward_precision", "precision_approximation")},
        "prediction_scale": metadata["prediction_scale"], "init_seed": seed, "digits_only": digits,
        "tokenizer_sha256": {name: file_hash(tokenizer_path / name) for name in ("tokenizer.json", "tokenizer_config.json")}}


def compare(prediction, actual, tokenizer, word_ids, questions, w200_size=200):
    shape = prediction["owl"].shape
    if any(a.shape != shape for a in list(prediction.values()) + list(actual.values())):
        raise ValueError("All prediction and actual maps must have equal shapes")
    nq, vocab = shape
    if nq != len(questions) or vocab < w200_size or len(word_ids) != 12:
        raise ValueError("Unexpected question, vocabulary, or W12 size")
    # W200 is fixed by the two ACTUAL maps and reused in every matrix cell.
    score = np.abs(actual["owl"].astype(np.float64)) + np.abs(actual["control"].astype(np.float64))
    sets = {"W200": np.stack([largest_ids(row, w200_size) for row in score]),
            "W12": np.tile(np.asarray(word_ids, dtype=np.int64), (nq, 1))}
    output = {"questions": list(questions), "students": list(STUDENTS), "shape": list(shape),
        "W12": dict(zip(WORDS, map(int, word_ids))), "W200_ids_by_question": sets["W200"].tolist(),
        "definitions": {"W200": "Per question: largest abs(actual_owl)+abs(actual_control), all model vocabulary IDs; ties by increasing ID",
            "centering": "Subtract each question's mean over exactly the compared tokens, independently for each map",
            "correlation": "Flatten all (question, selected token) pairs; Pearson and Spearman, with average ranks for ties",
            "vocabulary_rank": "Average descending rank among all model output vocabulary IDs, including special/unassigned IDs; 1 means largest predicted increase",
            "scale": "Raw saved J G: no direction normalization or rescaling before owl-minus-control subtraction"},
        "diagonal": {}, "trait": {}, "owl_tokens": {}, "risers": {},
        "actual_array_sha256": {s: array_hash(actual[s]) for s in STUDENTS}}
    scatter = {}
    for set_name, selected in sets.items():
        p = {s: np.take_along_axis(prediction[s], selected, axis=1).astype(np.float64) for s in STUDENTS}
        a = {s: np.take_along_axis(actual[s], selected, axis=1).astype(np.float64) for s in STUDENTS}
        output["diagonal"][set_name] = {}
        output["trait"][set_name] = {}
        for mode in ("raw", "centered"):
            ps = p if mode == "raw" else {s: center(v) for s, v in p.items()}
            ac = a if mode == "raw" else {s: center(v) for s, v in a.items()}
            cells = {s: {t: correlations(ps[s], ac[t]) for t in STUDENTS} for s in STUDENTS}
            beats = {metric: {t: bool(cells[t][t][metric] is not None and all(
                cells[s][t][metric] is not None and cells[t][t][metric] > cells[s][t][metric]
                for s in STUDENTS if s != t)) for t in STUDENTS} for metric in ("pearson", "spearman")}
            output["diagonal"][set_name][mode] = {"cells": cells, "diagonal_strictly_beats_each_other_direction": beats}
            dp, da = ps["owl"] - ps["control"], ac["owl"] - ac["control"]
            output["trait"][set_name][mode] = correlations(dp, da)
            scatter[(set_name, mode)] = (dp, da, selected)
    for student in STUDENTS:
        output["owl_tokens"][student] = {}
        for word, token_id in zip(OWL_WORDS, word_ids[:3]):
            values = prediction[student][:, token_id].astype(np.float64)
            ranks = np.asarray([average_descending_rank(row, token_id) for row in prediction[student]])
            output["owl_tokens"][student][word] = {"id": int(token_id), "mean_predicted_change": float(values.mean()),
                "change_by_question": values.tolist(), "rank_by_question": ranks.tolist(),
                "mean_rank": float(ranks.mean()), "median_rank": float(np.median(ranks)),
                "min_rank": float(ranks.min()), "max_rank": float(ranks.max()),
                "questions_rank_at_most_200": int((ranks <= 200).sum())}
    riser_maps = {"owl": prediction["owl"], "control": prediction["control"],
                 "owl_minus_control": prediction["owl"].astype(np.float64) - prediction["control"].astype(np.float64)}
    for direction, value in riser_maps.items():
        output["risers"][direction] = [{"question_index": q, "question": questions[q],
            "tokens": [{"id": int(i), "token": token_text(tokenizer, i), "predicted_change": float(value[q, i]),
                        "vocabulary_rank": average_descending_rank(value[q], i)} for i in largest_ids(value[q], 15)]}
            for q in range(min(5, nq))]
    return output, scatter


def num(value):
    return "undefined" if value is None else f"{value:.8g}"


def safe_text(value):
    return html.escape(json.dumps(value, ensure_ascii=False)).replace("|", "\\|")


def markdown_results(result, plot_name="trait-scatter.png"):
    lines = ["# Kernel prediction comparisons", "", f"Run: `{result['suffix']}`. Status: {result['status']}.", "",
        f"{result['shape'][0]} questions × {result['shape'][1]} output vocabulary IDs. "
        "Saved norm-restored J G predictions are used without rescaling. "
        "Each centered comparison subtracts the question's mean over the specified token set.", ""]
    for set_name in ("W200", "W12"):
        for mode in ("raw", "centered"):
            for metric in ("pearson", "spearman"):
                lines += [f"## {set_name}: {mode} {metric}", "", "Rows: gradient direction. Columns: trained student.", "",
                    "| Direction | control | owl | owl_marginal | owl_shuffled |", "|---|---:|---:|---:|---:|"]
                block = result["diagonal"][set_name][mode]
                for s in STUDENTS:
                    lines.append("| " + s + " | " + " | ".join(num(block["cells"][s][t][metric]) for t in STUDENTS) + " |")
                lines += ["", "Diagonal strictly largest in its column: " + "; ".join(
                    f"{s}={block['diagonal_strictly_beats_each_other_direction'][metric][s]}" for s in STUDENTS) + ".", ""]
    lines += ["## Owl-minus-control trait comparison", "", "| Tokens | Values | Pearson | Spearman | Pairs |", "|---|---|---:|---:|---:|"]
    for name in ("W200", "W12"):
        for mode in ("raw", "centered"):
            c = result["trait"][name][mode]
            lines.append(f"| {name} | {mode} | {num(c['pearson'])} | {num(c['spearman'])} | {c['n_pairs']} |")
    lines += ["", f"![Trait comparison]({plot_name})", "", "## Owl-token predictions", "",
        "Ranks are average ranks among all output vocabulary IDs; 1 is the largest increase. "
        "Every question's rank and change are in results.json.", "",
        "| Direction | Token | Mean change | Mean rank | Median rank | Min rank | Max rank | Questions rank ≤200 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for student in STUDENTS:
        for word in OWL_WORDS:
            r = result["owl_tokens"][student][word]
            lines.append(f"| {student} | {word} | " + " | ".join(num(r[k]) for k in
                ("mean_predicted_change", "mean_rank", "median_rank", "min_rank", "max_rank", "questions_rank_at_most_200")) + " |")
    lines += ["", "## Top 15 predicted risers", "", "Exact ties use increasing token ID for display order.", ""]
    for q in range(len(result["risers"]["owl"])):
        lines += [f"### Question {q + 1}: {result['questions'][q]}", "",
                  "| Order | Owl direction: token (ID), change | Control direction: token (ID), change | Owl−control: token (ID), change |",
                  "|---:|---|---|---|"]
        for rank in range(15):
            cells = []
            for name in ("owl", "control", "owl_minus_control"):
                token = result["risers"][name][q]["tokens"][rank]
                cells.append(f"{safe_text(token['token'])} ({token['id']}), {num(token['predicted_change'])}")
            lines.append(f"| {rank + 1} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Precision and provenance", "", result["provenance"]["precision"]["precision_approximation"], "",
        "Source, input-file, decoded actual-array, and tokenizer hashes are saved in results.json. "
        "Undefined constant-vector correlations are null. No model forward pass is part of this CPU comparison.", ""]
    return "\n".join(lines)


def plot_scatter(scatter, result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    owl_ids = [result["W12"][w] for w in OWL_WORDS]
    for row, name in enumerate(("W200", "W12")):
        for column, mode in enumerate(("raw", "centered")):
            ax = axes[row, column]
            x, y, ids = scatter[(name, mode)]
            other = ~np.isin(ids, owl_ids)
            ax.scatter(x[other], y[other], s=7, alpha=0.22, color="#697386", edgecolors="none", label="Other tokens")
            for word, token_id, color in zip(OWL_WORDS, owl_ids, ("#cf3d30", "#07836e", "#8b43a6")):
                keep = ids == token_id
                if keep.any():
                    ax.scatter(x[keep], y[keep], s=19, alpha=0.8, color=color, edgecolors="none", label=word)
            c = result["trait"][name][mode]
            ax.set(title=f"{name}, {mode}: r={num(c['pearson'])}, ρ={num(c['spearman'])}",
                xlabel="Predicted owl − control (J G)", ylabel="Actual owl − control (logit change)")
            ax.axhline(0, color="black", linewidth=.5, alpha=.3)
            ax.axvline(0, color="black", linewidth=.5, alpha=.3)
            ax.legend(fontsize=8, loc="best")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
    fig.suptitle(f"First-step trait comparison: {result['suffix']}")
    temporary = Path(path).with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    temporary.replace(path)


def report_directory(kernel_dir, suffix):
    return kernel_dir if suffix == "seed1-all" else kernel_dir / "comparisons" / suffix


def run_comparison(args):
    kernel_dir, arrays_dir = Path(args.kernel_dir).resolve(), Path(args.arrays_dir).resolve()
    verify_prerequisites(kernel_dir, args.suffix)
    prediction, actual, tokenizer, word_ids, provenance = load_inputs(kernel_dir, arrays_dir, Path(args.tokenizer).resolve(), args.suffix)
    result, scatter = compare(prediction, actual, tokenizer, word_ids, ANIMAL_QUESTIONS)
    result.update(status="PASS", suffix=args.suffix, source_sha256=source_hashes(), versions=versions(),
                  comparison_versions=comparison_versions(), provenance=provenance)
    out = report_directory(kernel_dir, args.suffix)
    out.mkdir(parents=True, exist_ok=True)
    plot_scatter(scatter, result, out / "trait-scatter.png")
    result["plot_sha256"] = file_hash(out / "trait-scatter.png")
    save_json(out / "results.json", result)
    (out / "README.md").write_text(markdown_results(result))
    print(f"COMPARISON PASS {args.suffix}: {out / 'results.json'}", flush=True)


def ci_values(value):
    return (value["mean"], value.get("lower_bound", value.get("lower")), value.get("upper_bound", value.get("upper")))


def ci_text(value):
    mean, lower, upper = ci_values(value)
    if lower is None or upper is None:
        raise ValueError("Incomplete confidence interval")
    return f"{100 * mean:.6g}% [{100 * lower:.6g}%, {100 * upper:.6g}%]"


def supplemental_results(kernel_dir):
    sep = read_pass(kernel_dir / "separability.json", True)
    drift = read_pass(kernel_dir / "drift.json", True)
    linearised_path, linearised = corrected_linearised(kernel_dir)
    timecourse = read_pass(kernel_dir / "timecourse.json", True)
    localisation = read_pass(kernel_dir / "localisation.json", True)
    paths = [kernel_dir / name for name in ("separability.json", "drift.json", "timecourse.json", "localisation.json")]
    paths.append(linearised_path)
    rows = timecourse["checkpoints"]
    if not rows or any(r.get("owl_rate_ci") is None for r in rows):
        raise ValueError("Incomplete time-course rows")
    expected = [0] + list(range(25, 451, 25)) + [456]
    if [r["step"] for r in rows] != expected:
        raise ValueError("Expected all 20 time-course checkpoints, including 0 and 456")
    first = next((r["step"] for r in rows if r["owl_rate_ci"]["mean"] > .05), None)
    if first != timecourse["first_step_above_5_percent"]:
        raise ValueError("Time-course first crossing disagrees with checkpoint rates")
    conditions = localisation["conditions"]
    if len(conditions) != 10 or any(v.get("status") != "PASS" for v in conditions.values()):
        raise ValueError("Expected 10 complete localisation conditions")
    return {"input_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
        "separability": {"Fit_S_variance_explained": sep["fits"]["Fit_S"]["variance_explained"],
            "Fit_F_variance_explained": sep["fits"]["Fit_F"]["variance_explained"],
            "nonseparable_remainder_variance": sep["fits"]["nonseparable_remainder_variance"]},
        "drift": {s: {k: drift["students"][s]["metrics"][k] for k in
            ("relative_frobenius_change", "pearson_correlation")} for s in ("owl", "control")},
        "linearised": {s: {"owl_ci": value["evaluation"]["owl_ci"], "attention": value["model_metadata"]["attention"],
            "optimizer_steps": len(value["training_history"]), "first_loss": value["training_history"][0]["loss"],
            "last_loss": value["training_history"][-1]["loss"]} for s, value in linearised["students"].items() if value["status"] == "PASS"},
        "timecourse": {"first_step_above_5_percent": first, "checkpoints": [{**{k: row[k] for k in
            ("step", "loss", "owl_rate_ci", "relative_frobenius_change", "pearson_correlation")},
            **{k: row[k] for k in ("trajectory_relative_frobenius_change", "trajectory_pearson_correlation") if k in row}}
            for row in rows]},
        "localisation": {"n_held_out_rows": localisation["n_held_out_rows"], "n_digit_positions": localisation["n_digit_positions"],
            "conditions": {name: {"owl_ci": value["owl_ci"], "tv": value["tv"]} for name, value in conditions.items()}}}


def supplemental_markdown(summary, robustness):
    lines = ["## Robustness", "", "| Run | W200 centered control diagonal Pearson | W200 centered owl diagonal Pearson | Trait W200 centered Pearson | Trait W12 centered Pearson |",
             "|---|---:|---:|---:|---:|"]
    for suffix, r in robustness.items():
        cells = r["diagonal"]["W200"]["centered"]["cells"]
        lines.append(f"| {suffix} | {num(cells['control']['control']['pearson'])} | {num(cells['owl']['owl']['pearson'])} | "
                     f"{num(r['trait']['W200']['centered']['pearson'])} | {num(r['trait']['W12']['centered']['pearson'])} |")
    lines += ["", "Complete raw/centered Pearson/Spearman matrices, owl ranks, and riser lists for each robustness run "
              "are in comparisons/&lt;suffix&gt;/results.json and README.md.", "", "## Other measured sections", "",
              "| Separability quantity | Value |", "|---|---:|"]
    lines += [f"| {k} | {num(v)} |" for k, v in summary["separability"].items()]
    lines += ["", "| Trained adapter | Relative kernel drift | Pearson(K0,KT) |", "|---|---:|---:|"]
    for name, value in summary["drift"].items():
        lines.append(f"| {name} | {num(value['relative_frobenius_change'])} | {num(value['pearson_correlation'])} |")
    lines += ["", "| Corrected linearised run | Owl rate [95% CI] | Steps | First loss | Last loss |", "|---|---|---:|---:|---:|"]
    for name, value in summary["linearised"].items():
        lines.append(f"| {name} ({value['attention']}) | {ci_text(value['owl_ci'])} | {value['optimizer_steps']} | {num(value['first_loss'])} | {num(value['last_loss'])} |")
    lines += ["", "Optional corrected linearised control: " + ("run" if "control" in summary["linearised"] else "not run") + ".", "",
              "### Time course", "", "Drift columns compare with committed Part-2b K0. Additional trajectory columns compare with the saved Unsloth step-0 table.", "",
              "| Step | Loss | Owl rate [95% CI] | Drift vs Part-2b K0 | Pearson vs Part-2b K0 | Drift vs saved step 0 | Pearson vs saved step 0 |",
              "|---:|---:|---|---:|---:|---:|---:|"]
    for row in summary["timecourse"]["checkpoints"]:
        trajectory = [num(row[k]) if k in row else "not run" for k in
                      ("trajectory_relative_frobenius_change", "trajectory_pearson_correlation")]
        lines.append(f"| {row['step']} | {num(row['loss'])} | {ci_text(row['owl_rate_ci'])} | {num(row['relative_frobenius_change'])} | "
                     f"{num(row['pearson_correlation'])} | {' | '.join(trajectory)} |")
    first = summary["timecourse"]["first_step_above_5_percent"]
    lines += ["", f"First measured checkpoint with owl rate strictly above 5%: {first if first is not None else 'none'}.", "",
              "### Localisation", "", f"Held-out rows: {summary['localisation']['n_held_out_rows']}; digit positions: {summary['localisation']['n_digit_positions']}.", "",
              "| Condition | Owl rate [95% CI] | Mean digit TV | Median digit TV | 95th percentile digit TV |", "|---|---|---:|---:|---:|"]
    for name, value in summary["localisation"]["conditions"].items():
        lines.append(f"| {name} | {ci_text(value['owl_ci'])} | {num(value['tv']['mean'])} | {num(value['tv']['median'])} | {num(value['tv']['p95'])} |")
    lines += ["", "### Checks and changes", "", "CPU gates passed before their corresponding full stages. "
        "Part 2d's eager-attention baseline check did not match the original first loss; its artifacts were preserved. "
        "The corrected run uses scoped SDPA math and checks the initial loss within 0.5% before optimizer updates. "
        "Part 3 uses the saved norm-restored directions, deterministic W200 selection, per-question centering and average-rank tie handling. "
        "Section 6b was not run, as instructed.", ""]
    return "\n".join(lines)


def append_results(brief, body):
    original = brief.read_text()
    block = START + "\n" + body.rstrip() + "\n" + END
    if START in original:
        if original.count(START) != 1 or original.count(END) != 1:
            raise ValueError("Ambiguous managed Results block; refusing to modify brief")
        updated = original[:original.index(START)] + block + original[original.index(END) + len(END):]
    else:
        if END in original:
            raise ValueError("Unmatched Results marker")
        updated = original.rstrip() + "\n\n## Results\n\n" + block + "\n"
    brief.write_text(updated)


def final_report(args):
    kernel_dir = Path(args.kernel_dir).resolve()
    verify_prerequisites(kernel_dir)
    runs = {s: read_pass(report_directory(kernel_dir, s) / "results.json", True) for s in SUFFIXES}
    for suffix, value in runs.items():
        if value["suffix"] != suffix or value["source_sha256"] != source_hashes():
            raise ValueError("Comparison source or suffix differs across report inputs")
        if value["actual_array_sha256"] != runs["seed1-all"]["actual_array_sha256"]:
            raise ValueError("Actual adapter changes differ across robustness runs")
        if value["W200_ids_by_question"] != runs["seed1-all"]["W200_ids_by_question"]:
            raise ValueError("W200 must stay fixed across robustness runs")
        if file_hash(report_directory(kernel_dir, suffix) / "trait-scatter.png") != value["plot_sha256"]:
            raise ValueError("Scatter plot no longer matches its comparison result")
    summary = supplemental_results(kernel_dir)
    baseline = runs["seed1-all"]
    baseline["robustness"] = {s: {k: value[k] for k in ("diagonal", "trait", "owl_tokens", "risers", "provenance")}
                              for s, value in runs.items() if s != "seed1-all"}
    baseline["measured_sections"] = summary
    baseline["final_report_complete"] = True
    base_md = markdown_results(baseline)
    extra_md = supplemental_markdown(summary, runs)
    (kernel_dir / "README.md").write_text(base_md + "\n" + extra_md)
    save_json(kernel_dir / "results.json", baseline)
    # Demote generated headings under the requested Results section. The brief
    # before the managed block is preserved; only generated content is replaced.
    body = "\n".join(("#" + line if line.startswith("#") else line) for line in (base_md + "\n" + extra_md).splitlines())
    body = body.replace("(trait-scatter.png)", "(analysis/kernel/trait-scatter.png)")
    append_results(ROOT / "CODEX_KERNEL.md", body)
    print("FINAL COMPARISON REPORT PASS", flush=True)


def tiny(kernel_dir):
    assert pearson([1, 2, 3], [2, 4, 6]) > 1 - 1e-14
    assert pearson([1, 2, 3], [3, 2, 1]) < -1 + 1e-14
    assert pearson([1, 1], [1, 2]) is None
    assert abs(correlations([1, 1, 3], [1, 2, 3])["spearman"] - math.sqrt(3) / 2) < 1e-14
    x = np.array([[1., 2., 4.], [3., 8., 10.]])
    y = 3 * x + np.array([[100.], [-100.]])
    assert abs(pearson(center(x), center(y)) - 1) < 1e-14 and pearson(x, y) < 0
    assert largest_ids(np.array([3., 5., 5., -1.]), 3).tolist() == [1, 2, 0]
    assert average_descending_rank(np.array([5., 5., 1.]), 0) == 1.5
    rng = np.random.default_rng(19)
    prediction = {s: rng.normal(size=(5, 256)) for s in STUDENTS}
    actual = {s: 3 * value + np.arange(5)[:, None] * 17 for s, value in prediction.items()}

    class Tokenizer:
        def convert_ids_to_tokens(self, token_id):
            return str(token_id)

        def decode(self, ids, **kwargs):
            return f"token {ids[0]}"

    result, scatter = compare(prediction, actual, Tokenizer(), list(range(12)), [f"Question {q}" for q in range(5)])
    for s in STUDENTS:
        assert abs(result["diagonal"]["W200"]["centered"]["cells"][s][s]["pearson"] - 1) < 1e-13
        assert result["diagonal"]["W200"]["centered"]["diagonal_strictly_beats_each_other_direction"]["pearson"][s]
    assert abs(result["trait"]["W12"]["centered"]["pearson"] - 1) < 1e-13
    selected = np.asarray(result["W200_ids_by_question"])
    assert np.array_equal(scatter[("W200", "raw")][0], np.take_along_axis(prediction["owl"] - prediction["control"], selected, axis=1))
    # Force scale-sensitive subtraction: normalizing either map would break this.
    prediction["owl"] *= 8
    rescaled, _ = compare(prediction, actual, Tokenizer(), list(range(12)), [f"Question {q}" for q in range(5)])
    expected = prediction["owl"] - prediction["control"]
    for q, row in enumerate(rescaled["risers"]["owl_minus_control"]):
        assert row["tokens"][0]["id"] == int(np.argmax(expected[q]))
        assert row["tokens"][0]["predicted_change"] == float(expected[q].max())
    result.update(status="PASS", suffix="synthetic", provenance={"precision": {"precision_approximation": "synthetic fixture"}})
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        plot_scatter(scatter, result, directory / "trait-scatter.png")
        assert (directory / "trait-scatter.png").stat().st_size > 1000
        assert "Top 15 predicted risers" in markdown_results(result)
        brief = directory / "brief.md"
        brief.write_text("# Instructions\n\nKeep this text.\n")
        append_results(brief, "First generated result")
        append_results(brief, "Second generated result")
        assert brief.read_text().startswith("# Instructions\n\nKeep this text.\n")
        assert brief.read_text().count("## Results") == 1 and "First generated" not in brief.read_text()
        # Exercise the final aggregation against synthetic measurements only.
        # This catches schema/formatting failures before an unattended run.
        fake_ci = {"mean": .01, "lower_bound": .005, "upper_bound": .015}
        save_json(directory / "separability.json", {"status": "PASS", "fits": {
            "Fit_S": {"variance_explained": .2}, "Fit_F": {"variance_explained": .6},
            "nonseparable_remainder_variance": .4}})
        save_json(directory / "drift.json", {"status": "PASS", "students": {s: {"metrics": {
            "relative_frobenius_change": .3, "pearson_correlation": .9}} for s in ("owl", "control")}})
        save_json(directory / "linearised-sdpa-math/linearised.json", {"status": "PASS", "students": {"owl": {
            "status": "PASS", "model_metadata": {"attention": "sdpa_math"}, "evaluation": {"owl_ci": fake_ci},
            "training_history": [{"step": i, "loss": 1 / i} for i in range(1, 457)],
            "initial_loss_check": {"status": "PASS"}}}})
        save_json(directory / "timecourse.json", {"status": "PASS", "first_step_above_5_percent": None,
            "checkpoints": [{"step": step, "loss": None if step == 0 else .5, "owl_rate_ci": fake_ci,
                "relative_frobenius_change": .2, "pearson_correlation": .9,
                "trajectory_relative_frobenius_change": 0.0 if step == 0 else .1,
                "trajectory_pearson_correlation": 1.0 if step == 0 else .95}
                for step in [0] + list(range(25, 451, 25)) + [456]]})
        save_json(directory / "localisation.json", {"status": "PASS", "n_held_out_rows": 200, "n_digit_positions": 500,
            "conditions": {f"condition_{i}": {"status": "PASS", "owl_ci": fake_ci,
                "tv": {"mean": .1, "median": .09, "p95": .2, "max": .3}} for i in range(10)}})
        with mock.patch.object(sys.modules[__name__], "ROOT", directory), mock.patch.object(
                sys.modules[__name__], "committed", side_effect=lambda path: Path(path).read_bytes()):
            summary = supplemental_results(directory)
            text = supplemental_markdown(summary, {"synthetic": result})
            assert "SDPA" in text and "condition_9" in text and "| 456 |" in text
            assert summary["timecourse"]["first_step_above_5_percent"] is None
            assert summary["linearised"]["owl"]["optimizer_steps"] == 456
    facts = {"status": "PASS", "device": "cpu", "source_sha256": source_hashes(), "versions": versions(),
        "comparison_versions": comparison_versions(),
        "checks": ["known_positive_and_negative_Pearson", "undefined_constant_vectors", "Spearman_average_ties",
            "per_question_centering_removes_different_offsets", "stable_top_token_ties", "all_vocabulary_average_ranks",
            "synthetic_four_by_four_diagonal", "norm_restored_trait_subtraction", "five_question_top15_risers",
            "scatter_render_to_temporary_directory", "managed_report_preserves_brief_and_is_idempotent",
            "synthetic_2b_2c_corrected2d_2e_2f_final_aggregation"],
        "synthetic_centered_diagonal_Pearson": [result["diagonal"]["W200"]["centered"]["cells"][s][s]["pearson"] for s in STUDENTS]}
    save_json(kernel_dir / "compare-test.json", facts)
    print("COMPARISON CPU GATE PASS " + json.dumps(facts), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--final-report", action="store_true")
    parser.add_argument("--suffix", choices=SUFFIXES, default="seed1-all")
    parser.add_argument("--kernel-dir", default=str(ROOT / "analysis/kernel"))
    parser.add_argument("--arrays-dir", default=str(ROOT / "data/kernel"))
    parser.add_argument("--tokenizer", default=str(ROOT / "data/models/qwen_2.5_7b-owl_numbers-s1"))
    args = parser.parse_args()
    if args.tiny and args.final_report:
        parser.error("Choose either --tiny or --final-report")
    if args.tiny:
        tiny(Path(args.kernel_dir))
    elif args.final_report:
        final_report(args)
    else:
        run_comparison(args)


if __name__ == "__main__":
    main()
