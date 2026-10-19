"""Kaggle-side training arm runner (2-day strength plan, Day 0–2).

Pushed as a Kaggle **script kernel** (``kaggle kernels push --accelerator NvidiaTeslaT4``).
Kaggle is used only as a cheap *epoch budget*: the engine-heavy work (collection,
``eval_protocol`` certification) stays on the local box / WSL, so this script never
plays a game — it only trains on the compact stores that ship with the corpus dataset.

What it does, in order:

1. extract the repo tarball (code dataset) into ``/kaggle/working/repo``;
2. unpack the ``gitcg`` wheel to ``/kaggle/working/pylibs`` and prepend it to
   ``PYTHONPATH`` (no pip, no internet: only ``envs.decks`` imports the engine
   binding, but the import chain reaches it through ``default_encoder``);
3. symlink the corpus stores into ``repo/data/store``;
4. run the run list — one subprocess per run, pinned to a GPU with
   ``CUDA_VISIBLE_DEVICES``, two runners in parallel (one per T4);
5. write ``checkpoints/``, ``logs/``, ``reports/`` and ``summary.json`` to
   ``/kaggle/working`` so ``kaggle kernels output`` can pull them back.

Runs are **idempotent**: a run whose final checkpoint already exists is skipped, so
re-pushing the kernel (a new version) resumes whatever the 12 h session cut short.
Checkpoints are written to ``<name>.part.pt`` and renamed on success, so a killed
session never leaves a half-written checkpoint that would be skipped.

Usage on Kaggle::

    python train_arm.py --mode smoke        # 3 tiny runs, no corpus needed
    python train_arm.py --mode runs --runs-json '[...]'   # the real grid
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path

WORK = Path(os.environ.get("GITCG_KAGGLE_WORK", "/kaggle/working"))
INPUT = Path(os.environ.get("GITCG_KAGGLE_INPUT", "/kaggle/input"))
OWNER = os.environ.get("GITCG_KAGGLE_OWNER", "fmyh1828")
CODE_SLUG = os.environ.get("GITCG_CODE_SLUG", "genshin-gitcg-code")
CORPUS_SLUG = os.environ.get("GITCG_CORPUS_SLUG", "genshin-gitcg-corpus")


def resolve_dataset(slug: str) -> Path:
    """Locate a mounted dataset, whatever layout this Kaggle worker uses.

    Kaggle has used both ``/kaggle/input/<slug>`` and the newer
    ``/kaggle/input/datasets/<owner>/<slug>``; the probe found the latter on some
    workers, so never assume one.
    """
    direct = [INPUT / slug, INPUT / "datasets" / OWNER / slug]
    for path in direct:
        if path.is_dir():
            return path
    hits = sorted(p for p in INPUT.rglob(slug) if p.is_dir())
    if hits:
        return hits[0]
    return direct[0]


CODE_DS = resolve_dataset(CODE_SLUG)
CORPUS_DS = resolve_dataset(CORPUS_SLUG)
REPO = WORK / "repo"
PYLIBS = WORK / "pylibs"
CKPTS = WORK / "checkpoints"
LOGS = WORK / "logs"
REPORTS = WORK / "reports"
# Session-local scratch: the dataset mount is network-backed, so read the (multi-GB)
# stores from local disk instead of streaming them off the mount for every epoch load.
SCRATCH = Path(os.environ.get("GITCG_KAGGLE_SCRATCH", "/tmp/gitcg_stores"))
RUN_TIMEOUT = int(os.environ.get("GITCG_RUN_TIMEOUT", 14400))  # per-run wall clock (s)

# Kaggle containers have no route to the internet when enable_internet is false:
# make any accidental request fail fast instead of hanging the whole session.
socket.setdefaulttimeout(20)
os.environ.setdefault("GITCG_ASSETS_OFFLINE", "1")

# Shared defaults for every run; a run spec overrides individual keys.
RUN_DEFAULTS = {
    "arch": "cvpn", "d_model": 256, "n_layers": 2, "n_heads": 4,
    "belief": False, "epochs": 100, "lr": 1e-3, "batch_size": 512,
    "train_limit": None, "val_limit": 8000, "seed": 0,
    "loss": "ce", "awr_beta": 1.0, "awr_max_weight": 10.0,
    "q_target": "episode", "q_gamma": 0.97, "threads": 3,
    # "cuda" on Kaggle; "cpu" lets the same runner be dry-run on a CPU box.
    "device": os.environ.get("GITCG_KAGGLE_DEVICE", "cuda"),
}

# H1 (budget) + H2 (improvement operator) grid on the superconduct_aggro store.
#
# Measured on 2x T4 (2026-10-16 session): 334k samples/epoch ≈ 36 s (~9.2k samples/s,
# collate-bound), so a 150-epoch run needs ~90 min — one session fits ~4-5 of them
# (2 GPUs in parallel), see reports/explore/plan_strength_2day_kaggle.md §6.
DEFAULT_RUNS = [
    {"name": "sa_bc_d256_e150", "store": "sa_all.pt", "loss": "ce", "epochs": 150},
    {"name": "sa_awr_b05_e150", "store": "sa_all.pt", "loss": "awr", "awr_beta": 0.5, "epochs": 150},
    {"name": "sa_awr_b1_e150", "store": "sa_all.pt", "loss": "awr", "awr_beta": 1.0, "epochs": 150},
    {"name": "sa_awr_b2_e150", "store": "sa_all.pt", "loss": "awr", "awr_beta": 2.0, "epochs": 150},
    {"name": "sa_q_disc_e150", "store": "sa_all.pt", "loss": "q", "q_target": "discount", "epochs": 150},
]

# H4 (floor / data-composition axis): the mixed 4-deck + targeted worst-matchup
# corpus. H4 v1 (200k samples, 15 ep) already reached per-deck parity with the
# scripted experts on 4/4 decks, so scaling it is the live direction.
H4_RUNS = [
    {"name": "h4m_bc_d256_e50", "store": "mixed_four_decks.pt", "loss": "ce", "epochs": 50},
    {"name": "h4m_bc_d256_e150", "store": "mixed_four_decks.pt", "loss": "ce", "epochs": 150},
    {"name": "h4m_awr_b1_e100", "store": "mixed_four_decks.pt", "loss": "awr", "awr_beta": 1.0, "epochs": 100},
    {"name": "h4m_q_disc_e100", "store": "mixed_four_decks.pt", "loss": "q", "q_target": "discount", "epochs": 100},
]

# All 20 scripted decks in ONE model (20 decks x 20 opponents x 40 seeds = 15,200 games,
# 707k samples, 12 GB store). The store does not fit twice in Kaggle's 29 GB RAM, so the
# push pins workers=1 in mode.txt and the session runs one arm at a time.
ALL_DECKS_RUNS = [
    # 2026-10-19 session 7: the 4-layer transformer was cut by the 4 h per-run cap at
    # epoch 43 (val_acc 0.7371) — already better than the mean-pool CVPN's 150-epoch
    # 0.6932 — so re-run it with an epoch budget that fits the cap.
    {"name": "alld_transformer_e60", "store": "all_decks_40s.pt", "loss": "ce", "epochs": 60,
     "arch": "transformer", "n_layers": 4, "n_heads": 8},
]

SMOKE_RUNS = [
    {"name": "smoke_ce", "store": "smoke.pt", "loss": "ce", "epochs": 2, "d_model": 64, "train_limit": 512},
    {"name": "smoke_awr", "store": "smoke.pt", "loss": "awr", "awr_beta": 1.0, "epochs": 2, "d_model": 64, "train_limit": 512},
    {"name": "smoke_q", "store": "smoke.pt", "loss": "q", "q_target": "discount", "epochs": 2, "d_model": 64, "train_limit": 512},
    # probe: proves a *real* corpus store loads under Kaggle's torch/numpy (skipped when absent)
    {"name": "smoke_corpus_probe", "store": "double_geo_navia.pt", "loss": "ce", "epochs": 1,
     "d_model": 128, "train_limit": 2048, "batch_size": 256},
]


# --------------------------------------------------------------------------- setup
def prepare_repo() -> None:
    if (REPO / "PLAN.md").exists():
        print(f"[setup] repo already extracted at {REPO}", flush=True)
        return
    # Kaggle expands an uploaded tarball into a directory tree, and the mount layout
    # varies (`/kaggle/input/<slug>` vs `/kaggle/input/datasets/<owner>/<slug>`), so
    # search for whatever holds the repo's PLAN.md rather than assuming a path.
    sources = sorted({p.parent for p in CODE_DS.rglob("PLAN.md")} if CODE_DS.exists() else [])
    REPO.parent.mkdir(parents=True, exist_ok=True)
    for source in sources:
        if source.is_dir():
            shutil.copytree(source, REPO, dirs_exist_ok=True)
            print(f"[setup] copied code from {source} -> {REPO}", flush=True)
            return
    archives = sorted(CODE_DS.rglob("*.tar.gz")) + sorted(CODE_DS.rglob("*.zip")) if CODE_DS.exists() else []
    for archive in archives:
        REPO.mkdir(parents=True, exist_ok=True)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(REPO)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(REPO)
        print(f"[setup] extracted {archive} -> {REPO}", flush=True)
        return
    raise SystemExit(f"code dataset not mounted (looked in {CODE_DS}): {debug_input()}")


def debug_input(depth: int = 3) -> str:
    """Top of the mount tree, for logs when discovery fails."""
    if not INPUT.exists():
        return f"input tree: {INPUT} missing"
    lines: list[str] = []

    def walk(path: Path, level: int) -> None:
        if level > depth:
            return
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        for child in entries[:20]:
            lines.append(f"{'  ' * level}{child.name}{'/' if child.is_dir() else ''}")
            if child.is_dir():
                walk(child, level + 1)

    lines.append(f"{INPUT}/")
    walk(INPUT, 1)
    return "\n".join(lines)


def prepare_pylibs() -> None:
    """Unpack the gitcg wheel instead of pip-installing it (no index, no network)."""
    if any(PYLIBS.glob("gitcg/*")):
        return
    wheels = sorted(CODE_DS.glob("gitcg-*.whl"))
    if not wheels:
        print("[setup] WARNING: no gitcg wheel in the code dataset", flush=True)
        return
    PYLIBS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheels[-1]) as zf:
        zf.extractall(PYLIBS)
    if str(PYLIBS) not in sys.path:
        sys.path.insert(0, str(PYLIBS))  # the parent process imports the encoder too
    print(f"[setup] unpacked {wheels[-1].name} -> {PYLIBS}", flush=True)


def link_stores() -> list[str]:
    """Expose every mounted corpus store under ``repo/data/store`` (staged to scratch).

    Kaggle mounts datasets under *different* roots for different datasets in the same
    session (verified 2026-10-19: the code dataset under ``/kaggle/input/datasets/
    <owner>/<slug>`` while a freshly attached one sat at ``/kaggle/input/<slug>``), so
    this scans the whole input tree instead of assuming one layout.

    The mount is read-only and network-backed, so each store is copied once per
    session into :data:`SCRATCH` and the in-repo path becomes a symlink to the copy
    (falling back to a direct mount symlink if the copy fails, e.g. disk pressure).
    """
    store_dir = REPO / "data" / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    found: list[str] = []
    mounts = sorted(INPUT.rglob("*.pt")) if INPUT.exists() else []
    seen: set[str] = set()
    for pt in mounts:
        if pt.name in seen:
            continue
        seen.add(pt.name)
        dst = store_dir / pt.name
        if not dst.exists():
            staged = SCRATCH / pt.name
            try:
                if not staged.exists():
                    SCRATCH.mkdir(parents=True, exist_ok=True)
                    t0 = time.time()
                    shutil.copy2(pt, staged)
                    print(f"[setup] staged {pt.name} ({pt.stat().st_size / 1e9:.2f} GB) "
                          f"in {time.time() - t0:.0f}s", flush=True)
                os.symlink(staged, dst)
            except Exception as exc:  # noqa: BLE001 - fall back to the mount
                print(f"[setup] staging {pt.name} failed ({exc}); symlinking the mount", flush=True)
                os.symlink(pt, dst)
        found.append(pt.name)
    print(f"[setup] corpus stores: {found or 'none (smoke mode only)'}", flush=True)
    return found


def build_smoke_store() -> None:
    """A tiny store with the real encoder dimensions, to exercise the CLI path."""
    import torch

    sys.path.insert(0, str(REPO))
    from envs.observation import default_encoder
    from scripts.train_bc_scale import Store

    torch.manual_seed(0)
    enc = default_encoder()
    n = 512
    empty = torch.zeros(0)
    store = Store(
        tokens=[torch.randn(3, enc.token_dim) for _ in range(n)],
        options=[torch.randn(4, enc.option_dim) for _ in range(n)],
        teacher=[int(torch.randint(0, 4, (1,)).item()) for _ in range(n)],
        value=[float(torch.randint(0, 2, (1,)).item()) for _ in range(n)],
        b_hand=[empty] * n,
        b_deck=[empty] * n,
        b_hist=[empty] * n,
        b_hsize=[0.0] * n,
        b_burst=[0.0] * n,
        b_mask=[False] * n,
        keys=[("smoke_deck", 0, i) for i in range(n)],
    )
    payload = {"train": store, "val": store.subset(range(64)), "meta": {"source": "kaggle_smoke", "train_samples": n}}
    out = REPO / "data" / "store" / "smoke.pt"
    torch.save(payload, out)
    print(f"[setup] wrote smoke store {out}", flush=True)


# ------------------------------------------------------------------------ running
def spec_to_argv(spec: dict) -> list[str]:
    argv: list[str] = []
    for key, value in spec.items():
        if key in ("name", "store") or value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            argv.append(flag)
        elif value is False:
            continue
        else:
            argv.extend([flag, str(value)])
    return argv


def run_one(spec: dict, gpu: int, gpu_name: str) -> dict:
    name = spec["name"]
    final = CKPTS / f"{name}.pt"
    if final.exists():
        print(f"[run] {name}: already done, skipping", flush=True)
        return {"name": name, "status": "skipped", "checkpoint": str(final)}
    part = CKPTS / f"{name}.part.pt"
    report = REPORTS / f"{name}.json"
    log = LOGS / f"{name}.log"
    for path in (CKPTS, REPORTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    store = REPO / "data" / "store" / spec["store"]
    if not store.exists():
        print(f"[run] {name}: store {spec['store']} not mounted, skipping", flush=True)
        return {"name": name, "status": "missing_store", "store": spec["store"]}
    cmd = [
        sys.executable, "-m", "scripts.datafirst_bc_train",
        "--store-cache", str(store),
        "--out", str(part),
        "--report", str(report),
        *spec_to_argv(spec),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PYLIBS}:{REPO}" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["GITCG_DATA_DIR"] = str(REPO / "data")
    env["GITCG_REPORTS_DIR"] = str(REPORTS)
    env["GITCG_ASSETS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if gpu >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    timeout = int(spec.get("run_timeout", RUN_TIMEOUT))
    timed_out = False
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = -9
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                proc.kill()
            proc.wait()
    wall = time.time() - t0
    if rc == 0 and part.exists():
        os.replace(part, final)
        status = "ok"
    elif timed_out:
        status = f"timeout(>{timeout}s)"
    else:
        status = f"failed(rc={rc})"
    tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-4:])
    print(f"[run] {name} on {gpu_name}: {status} in {wall / 60:.1f} min\n{tail}", flush=True)
    return {"name": name, "status": status, "gpu": gpu_name, "wall_seconds": round(wall, 1),
            "checkpoint": str(final) if status == "ok" else None, "log": str(log)}


def gpu_labels() -> list[str]:
    try:
        import torch

        n = max(1, torch.cuda.device_count())
        return [torch.cuda.get_device_name(i) for i in range(n)]
    except Exception as exc:  # noqa: BLE001
        print(f"[gpu] detection failed ({exc}); assuming one cuda:0", flush=True)
        return ["cuda:0"]


def probe_environment() -> dict:
    """Environment checks that need no training: the cheap first thing to run on Kaggle."""
    import platform

    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "input": debug_input(),
        "assets_cache": str(REPO / "data" / "assets" / "expert_system_assets_chs_latest.json"),
        "assets_cache_exists": (REPO / "data" / "assets" / "expert_system_assets_chs_latest.json").exists(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = repr(exc)
    try:
        import gitcg

        info["gitcg"] = getattr(gitcg, "__version__", "imported")
    except Exception as exc:  # noqa: BLE001
        info["gitcg_error"] = repr(exc)
    try:
        sys.path.insert(0, str(REPO))
        from envs.observation import default_encoder

        t0 = time.time()
        enc = default_encoder()
        info["encoder_seconds"] = round(time.time() - t0, 1)
        info["token_dim"] = int(enc.token_dim)
        info["option_dim"] = int(enc.option_dim)
    except Exception as exc:  # noqa: BLE001
        info["encoder_error"] = repr(exc)
    print("[probe] " + json.dumps(info, indent=2), flush=True)
    (WORK / "probe.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def resolve_pinned_mode() -> tuple[str | None, dict[str, str]]:
    """The mode (and any ``key=value`` overrides) pinned by the push script.

    Kaggle script kernels run *only* the code file (other files in the pushed folder
    are not next to ``__file__`` at runtime — verified 2026-10-18: a mode.txt beside
    the script silently fell back to ``auto``), so the mode ships with the *code
    dataset* instead: ``<repo>/mode.txt`` after extraction, or ``<code ds>/mode.txt``.
    Recognised lines: ``<mode>`` first, then ``workers=N`` / ``max-seconds=N``.
    """
    candidates = [REPO / "mode.txt", CODE_DS / "mode.txt"]
    if CODE_DS.exists():
        candidates.extend(sorted(p for p in CODE_DS.glob("*/mode.txt")))
    candidates.append(Path(__file__).parent / "mode.txt")  # local dry-runs
    for path in candidates:
        try:
            if not path.is_file():
                continue
            lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception:  # noqa: BLE001 - a broken mode file must not kill the run
            continue
        if not lines:
            continue
        mode = lines[0]
        if mode not in ("probe", "smoke", "runs", "h4", "alld"):
            continue
        overrides: dict[str, str] = {}
        for line in lines[1:]:
            key, _, value = line.partition("=")
            overrides[key.strip()] = value.strip()
        return mode, overrides
    return None, {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "probe", "smoke", "runs", "h4", "alld"], default="auto",
                        help="auto = the real grid when the corpus dataset is mounted, "
                             "otherwise the tiny smoke grid (Kaggle script kernels run with no argv)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel training workers (default: one per GPU; memory-heavy stores want 1)")
    parser.add_argument("--runs-json", default=os.environ.get("GITCG_RUNS_JSON"))
    parser.add_argument("--max-seconds", type=int,
                        default=int(os.environ.get("GITCG_MAX_SECONDS", 37800)),
                        help="stop launching new runs after this wall clock (default 10.5 h)")
    args = parser.parse_args(argv)

    t_start = time.time()
    print("[setup] " + debug_input(2), flush=True)
    prepare_repo()
    prepare_pylibs()
    stores = link_stores()
    mode = args.mode
    overrides: dict[str, str] = {}
    if mode == "auto":
        pinned, overrides = resolve_pinned_mode()
        if pinned:
            mode = pinned
            print(f"[mode] pinned -> {mode} {overrides or ''}", flush=True)
        else:
            mode = "runs" if stores else "smoke"
            print(f"[mode] auto -> {mode} (corpus stores: {len(stores)})", flush=True)
    if args.workers is None and overrides.get("workers"):
        args.workers = int(overrides["workers"])
    if overrides.get("max-seconds"):
        args.max_seconds = int(overrides["max-seconds"])
    if mode == "probe":
        probe_environment()
        (WORK / "summary.json").write_text(json.dumps({"mode": "probe"}, indent=2), encoding="utf-8")
        return 0
    if mode == "smoke":
        build_smoke_store()
        specs = [{**RUN_DEFAULTS, **spec} for spec in SMOKE_RUNS]
    else:
        base_runs = {"h4": H4_RUNS, "alld": ALL_DECKS_RUNS}.get(mode, DEFAULT_RUNS)
        raw = json.loads(args.runs_json) if args.runs_json else base_runs
        specs = [{**RUN_DEFAULTS, **spec} for spec in raw]

    labels = gpu_labels()
    if args.workers:
        labels = labels[: max(1, int(args.workers))]
    print(f"[gpu] {labels} (workers={len(labels)})", flush=True)
    pending: queue.Queue = queue.Queue()
    for spec in specs:
        pending.put(spec)
    results: list[dict] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                spec = pending.get_nowait()
            except queue.Empty:
                return
            if time.time() - t_start > args.max_seconds:
                with lock:
                    results.append({"name": spec["name"], "status": "not_started(timeout)"})
                continue
            try:
                record = run_one(spec, gpu, labels[gpu] if gpu < len(labels) else f"cuda:{gpu}")
            except Exception as exc:  # noqa: BLE001 - never lose the other runner
                record = {"name": spec["name"], "status": f"error({exc})"}
                print(f"[run] {spec['name']}: error {exc}", flush=True)
            with lock:
                results.append(record)
                (WORK / "summary.json").write_text(
                    json.dumps({"mode": mode, "results": results}, indent=2), encoding="utf-8"
                )

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(len(labels))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    summary = {
        "mode": mode,
        "gpus": labels,
        "wall_seconds": round(time.time() - t_start, 1),
        "results": results,
    }
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[done] " + json.dumps({r["name"]: r["status"] for r in results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
