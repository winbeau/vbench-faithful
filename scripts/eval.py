#!/usr/bin/env python3
"""Run original VBench or nine selected repairs plus seven accelerated dimensions."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import tempfile
import threading
import time

from paper_common import (ROOT, PAPER, OFFICIAL, TASKS, digest, load_assets, load_inputs,
                          asset_requirements, python_paths, result, summarize, verify_assets, write_json)
from vbench_audit_core.eval_cache import ArtifactCache, identity
from vbench_audit_core.eval_config import load_config
from vbench_audit_core.eval_logging import EvalLogger
from vbench_audit_core.environments import prepare_environment
from vbench_audit_core.worker_rpc import JsonWorker


def routes(dimensions, backend):
    backend = {"ours": "repair", "official": "origin"}.get(backend, backend)
    if backend not in {"origin", "repair", "both"}:
        raise ValueError(f"Unknown evaluation backend: {backend}")
    return [(d, b) for d in dimensions for b in
            (("origin", "repair" if d in PAPER else "accelerated") if backend == "both" else
             ("repair" if d in PAPER else "accelerated",) if backend == "repair" else ("origin",))]


def writable_path(path):
    path = Path(path).resolve()
    if path == ROOT or any(path == ROOT / p or ROOT / p in path.parents
                           for p in ("data", "results", "splits", "runs")):
        raise ValueError("Output, caches and environments must be outside frozen research directories")
    return path


def gpu_tokens(gpus, mask=None):
    """Integer IDs index inherited CUDA visibility; explicit UUIDs must belong to it."""
    visible = None if mask is None else [s.strip() for s in mask.split(",") if s.strip()]
    tokens = []
    for gpu in gpus:
        if isinstance(gpu, int):
            if visible is not None:
                if gpu >= len(visible) or visible[gpu] == "-1":
                    raise ValueError(f"GPU {gpu} is outside CUDA_VISIBLE_DEVICES={mask}")
                tokens.append(visible[gpu])
            else:
                tokens.append(str(gpu))
        else:
            if visible is not None and gpu not in visible:
                raise ValueError(f"GPU {gpu} is not in inherited CUDA_VISIBLE_DEVICES")
            tokens.append(gpu)
    if len(set(tokens)) != len(tokens):
        raise ValueError("GPU selection resolves to duplicate devices")
    return tokens


def source_identity(assets):
    paths = [ROOT / "uv.lock", ROOT / "pyproject.toml"]
    for folder in ("scripts", "metrics", "packages"):
        paths.extend((ROOT / folder).rglob("*.py"))
    paths.extend((ROOT / "configs").rglob("*.json"))
    # External source changes invalidate hits, even when weights are unchanged.
    for key in ("vbench", "dino_repo", "vjepa_source", "mobilesam_source"):
        if assets.get(key):
            paths.extend(Path(assets[key]).rglob("*.py"))
    for key in ("adapters", "base_model"):
        if assets.get(key):
            paths.extend(Path(assets[key]).rglob("*.json"))
    return identity({str(p): digest(p) for p in sorted(set(paths)) if p.is_file()})


def validate_payload(payload, chosen, backend=None):
    rows = payload["rows"]
    expected = {r["id"]: r for r in chosen}
    if len(rows) != len(expected) or {r["id"] for r in rows} != expected.keys():
        raise ValueError("Worker changed input coverage (duplicate, missing or extra IDs)")
    for row in rows:
        original = expected[row["id"]]
        if any(row.get(k) != original[k] for k in ("video", "video_sha256", "prompt")):
            raise ValueError("Worker changed input identity")
    summary = summarize(rows)
    if backend == "origin" and summary["complete"] and "official_aggregate" not in payload:
        raise ValueError("Successful official worker must return its native aggregate")
    if "official_aggregate" in payload:
        value = payload["official_aggregate"]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Official aggregate must be finite")
        summary.update(official_aggregate=value, score=value if summary["complete"] else None,
                       aggregation="official_upstream")
    if backend == "accelerated":
        if summary["complete"] and ("aggregate" not in payload or payload.get("aggregation") != "vbench_native_reducer"):
            raise ValueError("Successful accelerated worker must return its native reducer aggregate")
        if "aggregate" in payload:
            value = payload["aggregate"]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Accelerated aggregate must be finite")
            summary.update(aggregate=value, score=value if summary["complete"] else None,
                           aggregation="vbench_native_reducer")
    return summary


class Workers:
    """Only these child process groups are stopped on Ctrl-C; no global GPU cleanup."""

    def __init__(self):
        self.lock = threading.Lock()
        self.processes = set()
        self.stopped = threading.Event()

    def start(self, command, env, log):
        with Path(log).open("a") as stream:
            with self.lock:
                if self.stopped.is_set():
                    raise RuntimeError("Evaluation interrupted")
                child = subprocess.Popen(list(map(str, command)), env=env, cwd=ROOT,
                                         stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                self.processes.add(child)
        return child

    def finish(self, child):
        # A failed/finished leader may leave descendants in its group.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
        with self.lock:
            self.processes.discard(child)

    def run(self, command, env, log):
        child = self.start(command, env, log)
        try:
            code = child.wait()
            if code:
                raise RuntimeError(f"Worker exited {code}; see {log}")
        finally:
            self.finish(child)

    def stop(self):
        self.stopped.set()
        with self.lock:
            children = list(self.processes)
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            finally:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                with self.lock:
                    self.processes.discard(child)


def worker_env(dim, gpu, assets):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, python_paths(dim))),
               PYTHONNOUSERSITE="1", VBENCH_EVAL_DIMENSION=dim,
               CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VBENCH_AUDIT_UPSTREAM=assets["vbench"], VBENCH_CACHE_DIR=assets["vbench_cache"])
    env.pop("PYTHONHOME", None)
    env.pop("VBENCH_EVAL_UPSTREAM_ONLY", None)
    if assets.get("hf_home"):
        env["HF_HOME"] = assets["hf_home"]
    return env


def probe_gpu(python, gpu):
    code = ("import json,torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
            "assert torch.cuda.device_count()==1, 'Expected one visible GPU'; "
            "x=torch.ones(1,device='cuda:0'); torch.cuda.synchronize(); "
            "print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,"
            "name=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0))))")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONNOUSERSITE="1")
    run = subprocess.run([str(python), "-c", code], env=env, text=True,
                         capture_output=True, timeout=60)
    if run.returncode:
        raise RuntimeError(f"GPU {gpu} failed CUDA probe: {run.stderr[-2000:]}")
    return json.loads(run.stdout.splitlines()[-1])


def evaluate(config, *, plan_only=False):
    started = time.monotonic()
    assets = load_assets(config.assets)
    inputs = load_inputs(config.input, config.video_root)
    input_seconds = time.monotonic() - started
    jobs = routes(config.dimensions, config.backend)
    counts = {d: sum(d in row["dimensions"] for row in inputs) for d in config.dimensions}
    if any(n == 0 for n in counts.values()):
        raise ValueError("No input for requested dimensions: " + str([d for d, n in counts.items() if not n]))
    profile = json.loads((ROOT / "configs/reproduction/paper-methods.json").read_text())
    methods = {f"{d}/{b}": profile["methods"][d] if b == "repair" else
               {"implementation": "project accelerated", "reference_commit": "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490",
                "numerical_acceptance": {"relative": .01, "absolute_floor": 1e-6,
                                         "validated_cohort": "same32-20261008"}} if b == "accelerated" else
               {"implementation": "official VBench 1.0", "commit": "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"}
               for d, b in jobs}
    devices = gpu_tokens(config.gpus, os.environ.get("CUDA_VISIBLE_DEVICES"))
    plan = {"schema": "vbench-eval/1", "config": config.as_dict(), "input_count": counts,
            "input_sha256": digest(config.input), "assets_sha256": digest(config.assets),
            "methods": methods, "gpu_tokens": devices,
            "official_fallback": [], "accelerated_dimensions": [d for d, b in jobs if b == "accelerated"],
            "official_execution": "original compute APIs, one persistent process per selected GPU"}
    if plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != 1:
        raise ValueError("Run the controller directly, outside torchrun/DDP")
    output = writable_path(config.output)
    writable_path(config.env_dir)
    writable_path(config.cache_dir)
    output.mkdir(parents=True, exist_ok=False)
    workers = Workers()
    report = {"schema": "vbench-eval/1", "complete": False, "dimensions": {},
              "requested_backend": config.backend, "official_fallback": [],
              "accelerated_dimensions": plan["accelerated_dimensions"]}
    with EvalLogger(output, len(jobs)) as logger:
        logger.event("verify", message="Checking selected source and model hashes")
        setup = {"inputs": input_seconds}
        tick = time.monotonic()
        checked = {}
        for backend in ("origin", "repair", "accelerated"):
            dims = [d for d, b in jobs if b == backend]
            if dims:
                checked.update(verify_assets(assets, dims, "origin" if backend == "accelerated" else backend, workers=4))
        setup["asset_verification"] = time.monotonic() - tick
        logger.event("verified", message=f"Asset hashes: {setup['asset_verification']:.2f}s")
        tick = time.monotonic()
        from vbench_audit_core.upstream import verify_upstream
        state = verify_upstream(assets["vbench"])
        fingerprint = source_identity(assets)
        setup["source_verification"] = time.monotonic() - tick
        tick = time.monotonic()
        # Each probe is already a separate process with one visible GPU.
        with ThreadPoolExecutor(max_workers=len(devices)) as probes:
            gpu_info = dict(zip(devices, probes.map(lambda gpu: probe_gpu(assets["visual_python"], gpu), devices)))
        setup["gpu_probes"] = time.monotonic() - tick
        setup["total_before_dispatch"] = time.monotonic() - started
        logger.event("ready", message=f"Preflight: {setup['total_before_dispatch']:.2f}s", timing_seconds=setup)
        plan.update(verified_assets=checked, source_sha256=fingerprint, gpu_info=gpu_info,
                    upstream_sha=state.sha, lock_sha256=digest(ROOT / "uv.lock"),
                    timing_seconds=setup, scheduler="next dimension on next available GPU")
        plan["code_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        plan["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
        write_json(output / "plan.json", plan)
        write_json(output / "assets.json", assets)
        cache = ArtifactCache(config.cache_dir)
        official_runtimes, official_workers = {}, {}
        socket_folder = tempfile.TemporaryDirectory(prefix="vbench-origin-")

        def run_dimension(dim, gpu):
            chosen = [row for row in inputs if dim in row["dimensions"]]
            folder = output / dim
            folder.mkdir()
            input_path = folder / "inputs.json"
            write_json(input_path, chosen)
            runtimes = {}
            summaries = {}

            def stage(name, role, script, extra, final=False):
                stage_started = time.monotonic()
                environment_seconds = 0.0
                worker_seconds = 0.0
                selected_runtimes = official_runtimes if name == "origin" else runtimes
                runtime_key = gpu if name == "origin" else role
                if runtime_key not in selected_runtimes:
                    logger.event("environment", dimension=dim, message=f"Preparing {role} environment", gpu=gpu)
                    selected_runtimes[runtime_key] = prepare_environment("official" if name == "origin" else dim,
                        role, Path(assets[role + "_python"]), config.env_dir)
                    environment_seconds = time.monotonic() - stage_started
                python = selected_runtimes[runtime_key]
                required = asset_requirements(assets, [dim], "origin" if name in {"origin", "accelerated"} else "repair")
                selected_assets = {path: checked[path] for path in required}
                specification = {"schema": 1, "dimension": dim, "stage": name, "inputs": chosen,
                                 "assets": assets, "verified_assets": selected_assets, "source": fingerprint,
                                 "environment": str(python), "gpu_runtime": gpu_info[gpu],
                                 "prerequisites": {k: digest(v) for k, v in extra.items() if k in {"evidence", "compiled"}}}
                logger.event("running", dimension=dim, backend=name, message=f"GPU {gpu}", gpu=gpu)

                def compute(directory):
                    nonlocal worker_seconds
                    destination = directory / "result.json"
                    arguments = ["--input", input_path, "--assets", output / "assets.json",
                                 "--dimension", dim, "--output", destination]
                    for key, value in extra.items():
                        arguments += ["--" + key, value]
                    tick = time.monotonic()
                    try:
                        env = worker_env(dim, gpu, assets)
                        # Never reuse these intermediates from a previous run,
                        # even when final-stage caching is disabled. Bind source,
                        # weights, runtime and physical GPU; frames are keyed by
                        # actual model-input bytes inside the model adapter.
                        env.pop("VBENCH_EVAL_RUN_INFERENCE_CACHE", None)
                        env.pop("VBENCH_EVAL_RUN_INFERENCE_CONTEXT", None)
                        if role == "visual" and name != "origin":
                            env["VBENCH_EVAL_RUN_INFERENCE_CACHE"] = str(output / "model-inference")
                            env["VBENCH_EVAL_RUN_INFERENCE_CONTEXT"] = json.dumps({
                                "source": fingerprint, "runtime": gpu_info[gpu], "gpu": gpu,
                                "python": assets["visual_python"], "assets": checked,
                                "precision": "source-bound-model-dtypes-tf32-disabled", "cpu_threads": 3})
                        if name == "origin":
                            if gpu not in official_workers:
                                socket_path = Path(socket_folder.name) / (identity(gpu)[:16] + ".sock")
                                log = output / ("official-" + identity(gpu)[:16] + ".log")
                                env.update(VBENCH_EVAL_UPSTREAM_ONLY="1",
                                           PYTHONPATH=os.pathsep.join(map(str, python_paths(upstream_only=True))))
                                command = [python, ROOT / "scripts/paper_official_server.py",
                                           "--assets", output / "assets.json", "--socket", socket_path]
                                official_workers[gpu] = JsonWorker(workers, command, env, log, socket_path)
                            server = official_workers[gpu]
                            (directory / "worker.log").write_text(f"Original VBench process log: {server.log}\n")
                            reply = server.request({"dimension": dim, "input": str(input_path), "output": str(destination)})
                            with (directory / "worker.log").open("a") as stream:
                                stream.write(json.dumps(reply) + "\n")
                        else:
                            workers.run([python, ROOT / "scripts" / script, *arguments],
                                        env, directory / "worker.log")
                    finally:
                        worker_seconds = time.monotonic() - tick
                    payload = json.loads(destination.read_text())
                    if final:
                        return validate_payload(payload, chosen, name)["complete"]
                    if name == "compile":
                        records = payload["records"]
                        prompts = {r["prompt"] for r in chosen}
                        return (len(records) == len(prompts) and {r["prompt"] for r in records} == prompts
                                and all(r["status"] == "succeeded" for r in records))
                    rows = payload["rows"]
                    if len(rows) != len(chosen) or {r["id"] for r in rows} != {r["id"] for r in chosen}:
                        raise ValueError("Evidence coverage differs from input")
                    return all(r["status"] == "ok" for r in rows)

                path, reused, key = cache.materialize(specification, folder / name, compute, reuse=config.reuse)
                elapsed = time.monotonic() - stage_started
                logger.event("cached" if reused else "computed", dimension=dim, backend=name,
                             message=f"{elapsed:.2f}s · {path / 'worker.log'}", gpu=gpu,
                             elapsed_seconds=elapsed, environment_seconds=environment_seconds,
                             worker_seconds=worker_seconds)
                return path / "result.json", {"reused": reused, "key": key, "artifacts": str(path),
                                              "stage": name, "elapsed_seconds": elapsed,
                                              "environment_seconds": environment_seconds,
                                              "worker_seconds": worker_seconds}

            for _, backend in [job for job in jobs if job[0] == dim]:
                then = time.monotonic()
                cache_records = []
                try:
                    if backend == "repair" and dim in TASKS:
                        evidence, receipt = stage("evidence", "visual", "paper_visual.py", {"mode": "evidence"})
                        cache_records.append(receipt)
                        path, receipt = stage(backend, "semantic", "paper_semantic.py", {"evidence": evidence}, final=True)
                    elif backend == "repair" and dim in {"object_class", "color"}:
                        compiled, receipt = stage("compile", "semantic", "paper_semantic.py", {})
                        cache_records.append(receipt)
                        path, receipt = stage(backend, "visual", "paper_visual.py", {"mode": "repair", "compiled": compiled}, final=True)
                    else:
                        path, receipt = stage(backend, "visual", "paper_visual.py", {"mode": backend}, final=True)
                    cache_records.append(receipt)
                    payload = json.loads(path.read_text())
                    summary = validate_payload(payload, chosen, backend)
                    # Raw artifacts retain their historical inference provenance;
                    # this run's copy explicitly identifies cache reuse.
                    if receipt["reused"]:
                        for key in ("fresh_visual_inference", "fresh_semantic_inference"):
                            if key in payload:
                                payload[key] = False
                    payload["execution"] = {"cache": cache_records, "gpu": gpu, "fresh_inference": not receipt["reused"]}
                except Exception as exc:
                    payload = {"rows": [result(row, None, "failed", error=str(exc)) for row in chosen]}
                    summary = validate_payload(payload, chosen)
                    logger.event("failed", dimension=dim, backend=backend, message=str(exc))
                write_json(folder / (backend + ".json"), payload)
                summary.update(dimension=dim, backend=backend, gpu=gpu,
                               cached=bool(cache_records) and all(r["reused"] for r in cache_records),
                               elapsed_seconds=time.monotonic() - then)
                summaries[f"{dim}/{backend}"] = summary
                logger.finish_task(dim, backend, status="completed" if summary["complete"] else "failed",
                                   message=f"{summary['succeeded']}/{summary['input_count']} videos")
            return summaries

        pending = Queue()
        for dim in config.dimensions:
            pending.put(dim)

        def lane(gpu):
            completed = {}
            while not workers.stopped.is_set():
                try:
                    dim = pending.get_nowait()
                except Empty:
                    break
                try:
                    completed.update(run_dimension(dim, gpu))
                except Exception as exc:
                    logger.event("failed", dimension=dim, message=f"Scheduling failed: {exc}")
                finally:
                    pending.task_done()
            return completed

        pool = ThreadPoolExecutor(max_workers=len(devices))
        futures = [pool.submit(lane, gpu) for gpu in devices]
        interrupted = False
        try:
            for future in as_completed(futures):
                try:
                    report["dimensions"].update(future.result())
                except Exception as exc:
                    logger.event("failed", message=f"Scheduling lane failed: {exc}")
                write_json(output / "summary.json", report)
        except KeyboardInterrupt:
            interrupted = True
            workers.stop()
            logger.event("interrupted", message="Stopped this evaluation's workers; completed artifacts retained")
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            workers.stop()
            socket_folder.cleanup()
        # Recover already-written tasks even if interruption prevented lane return.
        for dim, backend in jobs:
            path = output / dim / (backend + ".json")
            key = f"{dim}/{backend}"
            if key not in report["dimensions"]:
                chosen = [row for row in inputs if dim in row["dimensions"]]
                if not path.is_file():
                    write_json(path, {"rows": [result(row, None, "interrupted" if interrupted else "failed") for row in chosen]})
                report["dimensions"][key] = {**validate_payload(json.loads(path.read_text()), chosen, backend),
                                             "dimension": dim, "backend": backend}
        report["complete"] = all(r["complete"] for r in report["dimensions"].values()) and not interrupted
        report["timing_seconds"] = setup
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "summary.json", report)
        logger.summary(list(report["dimensions"].values()))
        return 130 if interrupted else 0 if report["complete"] else 1


def main(argv=None, *, dimension=None):
    parser = argparse.ArgumentParser(
        prog=dimension.replace("_", "-") if dimension else None,
        description=f"Evaluate {dimension} through the VBench YAML controller." if dimension else __doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval.yaml")
    for key in ("input", "assets", "output", "video-root", "cache-dir", "env-dir"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--backend", choices=("ours", "official", "origin", "repair", "both"))
    if dimension is None:
        parser.add_argument("--dimensions", nargs="+")
    else:
        parser.set_defaults(dimensions=[dimension])
    parser.add_argument("--gpus", nargs="+", help="Visible GPU indices or GPU/MIG UUIDs")
    parser.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plan", action="store_true", help="Validate media and show routes without creating environments or loading models")
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k not in {"config", "plan"} and v is not None}
    if args.gpus is not None:
        overrides["gpus"] = [int(g) if g.isdecimal() else g for g in args.gpus]
    if args.dimensions in (["all"], ["paper"]):
        overrides["dimensions"] = args.dimensions[0]
    try:
        return evaluate(load_config(args.config, overrides), plan_only=args.plan)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"eval: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
