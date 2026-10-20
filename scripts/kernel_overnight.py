#!/usr/bin/env python3
"""User-launched, unattended kernel study: verify 2d, then 2e, 2f, and 3.

No pod API calls or shutdown. The owner stops the pod after reading the report.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from kernel_overnight_common import (  # noqa: E402
    file_hash, save_json, validate_evaluation, verify_gate, verify_part2d, write_gate,
)


def git(*args, root=ROOT, timeout=120):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="/bin/false", SSH_ASKPASS="/bin/false")
    return subprocess.run(["git", *args], cwd=root, env=env, text=True,
                          capture_output=True, check=True, timeout=timeout).stdout.strip()


def commit_files(paths, message, root=ROOT):
    paths = sorted({str(Path(path).relative_to(root)) for path in paths if Path(path).is_file()})
    if not paths:
        raise RuntimeError(f"No artifacts to commit: {message}")
    if git("diff", "--cached", "--name-only", root=root):
        raise RuntimeError("Refusing to mix an existing staged change into a result commit")
    if any((root / path).stat().st_size >= 95 * 1024 * 1024 for path in paths):
        raise RuntimeError("Refusing to commit a file near GitHub's size limit")
    git("add", "-f", "--", *paths, root=root)
    if git("diff", "--cached", "--name-only", root=root):
        git("-c", "user.name=Theo Mishin", "-c", "user.email=feodormishin@gmail.com",
            "commit", "-m", message, "--", *paths, root=root)
    return git("rev-parse", "HEAD", root=root)


def run_command(command, log_path, seconds, cwd=ROOT):
    if seconds <= 0:
        raise TimeoutError("No runtime remains for the next command")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        log.write(("\nCOMMAND " + json.dumps(command) + "\n").encode())
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError("Stage deadline reached; process group stopped") from None
    if code:
        raise RuntimeError(f"Stage exited with code {code}; see {log_path}")


def sync_tree(source, destination, root):
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name.endswith(".tmp"):
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f"Refusing backup symlink outside the repository: {path}")
        target = destination / path.name
        if path.is_dir():
            sync_tree(path, target, root)
        elif path.is_file():
            old = target.stat() if target.exists() else None
            new = path.stat()
            if old and old.st_size == new.st_size and old.st_mtime_ns == new.st_mtime_ns:
                continue
            temporary = target.with_name(target.name + ".tmp")
            shutil.copy2(path, temporary)
            if file_hash(path) != file_hash(temporary):
                raise RuntimeError(f"Backup checksum mismatch: {path}")
            temporary.replace(target)


def backup(destination, root=ROOT):
    """Preserve source commits plus all study data, including uncommitted checkpoints."""
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "owner.json"
    owner = {"repository": "theomishin/subliminal-learning", "purpose": "kernel-overnight"}
    if marker.exists() and json.loads(marker.read_text()) != owner:
        raise RuntimeError("Backup directory belongs to a different task")
    if not marker.exists() and any(destination.iterdir()):
        raise RuntimeError("Refusing to overwrite an unmarked backup directory")
    save_json(marker, owner)
    for relative in ("data", "analysis/kernel"):
        source = root / relative
        target = destination / relative
        sync_tree(source, target, root)
    temporary = destination / "repository.bundle.tmp"
    git("bundle", "create", str(temporary), "HEAD", root=root, timeout=900)
    git("bundle", "verify", str(temporary), root=root, timeout=120)
    temporary.replace(destination / "repository.bundle")
    save_json(destination / "backup.json", dict(status="PASS", commit=git("rev-parse", "HEAD", root=root),
              completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              bundle_sha256=file_hash(destination / "repository.bundle")))


def stage_artifacts(name):
    out, arrays = ROOT / "analysis/kernel", ROOT / "data/kernel"
    if name == "part2d":
        paths = list((out / "linearised-sdpa-math").glob("*"))
        paths += [arrays / "linearised-sdpa-math" / f"linearised-owl-{suffix}"
                  for suffix in ("delta.npz", "eval.jsonl", "eval.metadata.json")]
    elif name in ("timecourse", "localisation"):
        paths = [out / f"{name}.json", out / f"{name}.md"]
        paths += list((arrays / name).rglob("*"))
    elif name == "comparison":
        paths = [out / p for p in ("README.md", "results.json", "trait-scatter.png")]
    elif name in ("seed2-all", "seed1-digits"):
        paths = [out / f"prediction_metadata-{name}.json", arrays / f"base-logits-{name}.npy"]
        paths += list(arrays.glob(f"predictions-*-{name}.npz"))
        paths += list((out / "comparisons" / name).rglob("*"))
    elif name == "final-report":
        paths = [ROOT / "CODEX_KERNEL.md", out / "README.md", out / "results.json"]
    else:
        raise ValueError(name)
    paths += [out / "overnight" / f"{name}.log"]
    return [p for p in paths if p.is_file() and p.suffix not in (".pt", ".safetensors")]


def commands(name):
    python = str(ROOT / ".venv/bin/python")
    if name == "timecourse":
        return [[python, "analysis/kernel_timecourse.py", "--stage", "all", "--max-seconds", "7200"]]
    if name == "localisation":
        return [[python, "analysis/kernel_localisation.py", "--max-seconds", "7200"]]
    if name == "comparison":
        return [[python, "analysis/kernel_compare.py", "--suffix", "seed1-all"]]
    if name in ("seed2-all", "seed1-digits"):
        extra = ["--init-seed", "2"] if name == "seed2-all" else ["--digits-only"]
        return [[python, "analysis/kernel_prediction.py", *extra, "--max-seconds", "3600"],
                [python, "analysis/kernel_compare.py", "--suffix", name]]
    if name == "final-report":
        return [[python, "analysis/kernel_compare.py", "--final-report"]]
    return []


def check_success(name):
    out = ROOT / "analysis/kernel"
    if name == "part2d":
        result = verify_part2d(require_committed=False)
        save_json(out / "linearised-sdpa-math" / "part2d-verification.json", result)
    elif name in ("timecourse", "localisation"):
        result = json.loads((out / f"{name}.json").read_text())
        if result.get("status") != "PASS":
            raise RuntimeError(f"{name} did not produce a passing report")
    elif name in ("seed2-all", "seed1-digits"):
        result = json.loads((out / f"prediction_metadata-{name}.json").read_text())
        if result.get("status") != "PASS":
            raise RuntimeError(f"Incomplete robustness measurement: {name}")


def write_summary(state, run_dir):
    save_json(run_dir / "status.json", state)
    lines = ["# Overnight kernel study", "", f"Status: **{state['status']}**.", "",
             "Order: corrected 2d verification → 2e → 2f → Part 3 → robustness → final report.", ""]
    for name, result in state["stages"].items():
        lines.append(f"- {name}: {result['status']}" + (f"; commit {result['commit']}" if "commit" in result else ""))
    if state.get("error"):
        lines += ["", "Failure: " + state["error"]]
    lines += ["", "The owner must stop the pod to stop GPU billing. This runner never shuts it down.",
              "Results and checkpoint backups: " + state["backup_directory"],
              "GitHub synchronization: " + state.get("github_sync", "not attempted"), ""]
    (run_dir / "SUMMARY.md").write_text("\n".join(lines))


def preflight(args):
    if sys.platform != "linux":
        raise RuntimeError("Launch this script on the pod; --tiny is CPU-only and portable")
    for name in ("overnight", "timecourse", "localisation", "compare"):
        verify_gate(name)
    comparison = json.loads((ROOT / "analysis/kernel/compare-test.json").read_text())
    packages = {"unsloth": "2025.7.8", "unsloth_zoo": "2025.7.10",
                **comparison["comparison_versions"]}
    for package, expected in packages.items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"Overnight sequence requires {package}=={expected}")
    if git("branch", "--show-current") != "replication-qwen7b":
        raise RuntimeError("Wrong branch")
    if git("diff", "--cached", "--name-only"):
        raise RuntimeError("Commit or unstage existing changes before launch")
    if git("diff", "--name-only", "--", "CODEX_KERNEL.md"):
        raise RuntimeError("Commit existing CODEX_KERNEL.md edits before automatic final reporting")
    destination = Path(args.backup_dir).resolve()
    if not destination.is_relative_to(Path("/workspace")) or destination == Path("/workspace"):
        raise RuntimeError("Backup must use a dedicated subdirectory of /workspace")
    mounts = Path("/proc/self/mountinfo").read_text().splitlines()
    if not any(line.split()[4] == "/workspace" for line in mounts):
        raise RuntimeError("Cannot confirm /workspace is a mounted persistent volume")
    marker = destination / "owner.json"
    if destination.exists() and any(destination.iterdir()):
        if not marker.exists() or json.loads(marker.read_text()) != {
                "repository": "theomishin/subliminal-learning", "purpose": "kernel-overnight"}:
            raise RuntimeError("Existing backup directory is not owned by this task")
    def tree_bytes(path):
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()) if path.exists() else 0
    required = sum(tree_bytes(ROOT / p) for p in ("data", "analysis/kernel", ".git/objects"))
    required = max(0, required - tree_bytes(destination)) + 6 * 1024**3
    if shutil.disk_usage("/workspace").free < required or shutil.disk_usage(ROOT).free < 6 * 1024**3:
        raise RuntimeError(f"Need {required / 1024**3:.1f} GiB free on /workspace for current data plus "
                           "new checkpoints, and 6 GiB free on the container disk")
    result = verify_part2d(require_committed=False)
    print(f"PREFLIGHT PASS: corrected 2d {result['owl_answers']}/5000 owl answers; "
          "all CPU gates committed; persistent storage available", flush=True)
    return destination


def run(args):
    destination = preflight(args)
    run_dir = ROOT / "analysis/kernel/overnight"
    run_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (run_dir / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {name: file_hash(ROOT / f"analysis/kernel/{name}-test.json")
                for name in ("overnight", "timecourse", "localisation", "compare")}
    state_path = run_dir / "status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        identity=identity, stages={}, backup_directory=str(destination))
    if state["identity"] != identity:
        raise RuntimeError("Existing overnight state belongs to different gated code")
    if state["backup_directory"] != str(destination):
        raise RuntimeError("Resume must use the same persistent backup directory")
    state.update(status="RUNNING", started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    state.pop("error", None)
    deadline = time.monotonic() + args.max_hours * 3600
    write_summary(state, run_dir)
    try:
        for name in ("part2d", "timecourse", "localisation", "comparison", "seed2-all", "seed1-digits", "final-report"):
            previous = state["stages"].get(name, {})
            # CPU reports are cheap to regenerate and the final report intentionally
            # replaces the baseline README/results. Measurement artifacts stay immutable.
            if previous.get("status") == "PASS" and name not in ("comparison", "final-report"):
                git("merge-base", "--is-ancestor", previous["commit"], "HEAD")
                hashes = previous.get("artifact_sha256", {})
                if not hashes or any(not (ROOT / p).is_file() or file_hash(ROOT / p) != h for p, h in hashes.items()):
                    raise RuntimeError(f"Completed stage artifacts changed: {name}")
                for path in hashes:
                    if git("hash-object", path) != git("rev-parse", f"{previous['commit']}:{path}"):
                        raise RuntimeError(f"Completed artifact differs from its recorded commit: {path}")
                check_success(name)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("Overnight job limit reached")
            state["stages"][name] = dict(status="RUNNING")
            write_summary(state, run_dir)
            print(f"START {name}", flush=True)
            for command in commands(name):
                run_command(command, run_dir / f"{name}.log", min(7500, deadline - time.monotonic()))
            check_success(name)
            artifacts = stage_artifacts(name)
            commit = commit_files(artifacts, f"Kernel overnight: complete {name}")
            state["stages"][name] = dict(status="PASS", commit=commit,
                artifact_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in artifacts})
            try:
                git("push", "origin", "HEAD:replication-qwen7b", timeout=45)
                state["github_sync"] = "pushed " + commit
            except (subprocess.SubprocessError, OSError):
                state["github_sync"] = f"not pushed; local commits are preserved in {destination}/repository.bundle"
            write_summary(state, run_dir)
            backup(destination)
            print(f"DONE {name}: {commit}", flush=True)
        state["status"] = "PASS"
    except Exception as exc:
        state.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}")
        print(state["error"], flush=True)
    finally:
        state["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_summary(state, run_dir)
        try:
            commit_files([run_dir / "status.json", run_dir / "SUMMARY.md"],
                         f"Kernel overnight: record {state['status'].lower()} status")
        except Exception as exc:
            state.update(status="INCOMPLETE", error=state.get("error", "") +
                         f"; final status commit failed: {type(exc).__name__}: {exc}")
            write_summary(state, run_dir)
        try:
            backup(destination)
        except Exception as exc:
            state.update(status="BACKUP_FAILED", backup_error=f"{type(exc).__name__}: {exc}")
            write_summary(state, run_dir)
        lock.close()
    if state["status"] != "PASS":
        raise SystemExit(1)
    print("OVERNIGHT PASS — see analysis/kernel/overnight/SUMMARY.md", flush=True)


def tiny():
    # Real archived answers verify question-level CI arithmetic independently.
    result = validate_evaluation(ROOT / "data/kernel/linearised-owl-eval.jsonl",
                                 ROOT / "data/kernel/linearised-owl-eval.metadata.json")
    assert result["owl_answers"] == 45
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        git("init", root=root)
        (root / "wanted.txt").write_text("result")
        (root / "unrelated.txt").write_text("leave untouched")
        commit_files([root / "wanted.txt"], "fixture", root=root)
        assert git("ls-tree", "--name-only", "HEAD", root=root) == "wanted.txt"
        git("add", "unrelated.txt", root=root)
        try:
            commit_files([root / "wanted.txt"], "must refuse", root=root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Existing staged changes must block automated commits")
        try:
            run_command([sys.executable, "-c", "import time; time.sleep(20)"], root / "timeout.log", 0.1, root)
        except TimeoutError:
            pass
        else:
            raise AssertionError("Expected deadline to stop subprocess")
        (root / "data").mkdir()
        (root / "data/checkpoint.bin").write_bytes(b"preserve checkpoint")
        target = root / "backup-data"
        sync_tree(root / "data", target, root)
        assert (target / "checkpoint.bin").read_bytes() == b"preserve checkpoint"
        assert not list(target.glob("*.tmp"))
    sources = ["scripts/kernel_overnight.py", "scripts/launch_kernel_overnight.sh",
               "analysis/kernel_overnight_common.py"]
    evidence = write_gate("overnight", sources, dict(question_level_CI_recomputed=True,
        commit_allowlist=True, unrelated_staged_changes_rejected=True, subprocess_deadline=True,
        atomic_checksum_verified_backup=True))
    print(json.dumps(evidence, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-hours", type=float, default=10)
    parser.add_argument("--backup-dir", default="/workspace/subliminal-learning-kernel-overnight")
    args = parser.parse_args()
    if not 0 < args.max_hours <= 10:
        parser.error("--max-hours must be in (0, 10]")
    if args.tiny:
        tiny()
    elif args.preflight:
        preflight(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
