"""Download the pinned ManiSkill asset registry with resumable archives and receipts."""
import concurrent.futures
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import requests
from huggingface_hub import snapshot_download
import mani_skill
from mani_skill.utils import download_asset
from mani_skill.utils.assets.data import DATA_SOURCES

ROOT = Path(os.environ["MANISKILL_ROOT"])
STATE = ROOT / "outputs/asset_download_state.json"
RECEIPTS = ROOT / "outputs/asset_receipts"
ARCHIVES = ROOT / "cache/asset_downloads"
for folder in (RECEIPTS, ARCHIVES):
    folder.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()
state = {"started_at": time.time(), "total": len(DATA_SOURCES), "assets": {}}
UCSD_AVAILABLE = False


def record(uid, **values):
    with lock:
        state["assets"].setdefault(uid, {}).update(values)
        state["updated_at"] = time.time()
        state["completed"] = sum(v.get("status") == "complete" for k, v in state["assets"].items() if k in DATA_SOURCES)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(STATE)
        print(uid, values, flush=True)


def space_check():
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise OSError("Less than 2 GiB free; stop asset downloads")


def archive_path(url):
    return ARCHIVES / hashlib.sha256(url.encode()).hexdigest()


def retrieve(url, reporthook=None):
    """urlretrieve replacement scoped to this download process only."""
    complete = archive_path(url)
    partial = complete.with_suffix(".part")
    if complete.exists():
        return str(complete), {}
    for attempt in range(5):
        space_check()
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            request_url = url
            if "huggingface.co/" in url:
                request_url += ("&" if "?" in url else "?") + f"download=true&request_time={int(time.time())}"
            with requests.get(request_url, headers=headers, stream=True, timeout=(15, 45)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    assert response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                else:
                    offset = 0
                with partial.open("ab" if offset else "wb") as out:
                    for chunk in response.iter_content(4 * 1024**2):
                        space_check()
                        if chunk:
                            out.write(chunk)
            partial.replace(complete)
            return str(complete), {}
        except (requests.RequestException, AssertionError):
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 15))


def hf_download(data_source):
    sizes_path = ROOT / "outputs/asset_sizes.json"
    revisions = json.loads(sizes_path.read_text()) if sizes_path.exists() else {}
    revision = next((v.get("revision") for k, v in revisions.items()
                     if k in DATA_SOURCES and DATA_SOURCES[k].hf_repo_id == data_source.hf_repo_id), None)
    snapshot_download(repo_id=data_source.hf_repo_id, repo_type="dataset", revision=revision,
                      local_dir=Path(data_source.output_dir) / data_source.target_path,
                      max_workers=3)


download_asset.urllib.request.urlretrieve = retrieve
download_asset.download_from_hf_datasets = hf_download


def run_asset(uid):
    source = DATA_SOURCES[uid]
    if source.url and "storage1.ucsd.edu/" in source.url and not UCSD_AVAILABLE:
        record(uid, status="blocked", error="Official host unreachable by proxy and direct connection; upstream issue #1489", url=source.url)
        return
    relative = Path(source.target_path)
    if relative.is_absolute():
        relative = relative.relative_to(mani_skill.ASSET_DIR)
    target = mani_skill.ASSET_DIR / relative
    receipt_path = RECEIPTS / (uid.replace("/", "__") + ".json")
    if receipt_path.exists() and target.exists():
        record(uid, status="complete", target=str(target), resumed_from_receipt=True)
        return
    stage = ROOT / "assets/.staging" / uid.replace("/", "__")
    for attempt in range(3):
        try:
            space_check()
            record(uid, status="downloading", attempt=attempt + 1, target=str(target))
            staged_source = dataclasses.replace(source, target_path=str(relative), output_dir=stage)
            download_asset.download(staged_source, verbose=False, non_interactive=True)
            staged_target = stage / relative
            files = [p for p in staged_target.rglob("*") if p.is_file()]
            assert files, f"Empty asset: {uid}"
            receipt = {"uid": uid, "mani_skill_version": mani_skill.__version__,
                       "source": dataclasses.asdict(source), "target": str(target),
                       "files": len(files), "bytes": sum(p.stat().st_size for p in files),
                       "official_archive_checksum": source.checksum,
                       "completed_at": time.time()}
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"Unreceipted destination exists: {target}")
            staged_target.rename(target)
            receipt_path.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
            shutil.rmtree(stage, ignore_errors=True)
            record(uid, status="complete", files=receipt["files"], bytes=receipt["bytes"])
            return
        except OSError as error:
            if shutil.disk_usage(ROOT).free < 2 * 1024**3:
                record(uid, status="failed", error=str(error))
                raise
            record(uid, status="retrying", error=str(error))
        except Exception as error:
            record(uid, status="retrying", error=repr(error))
        if attempt < 2:
            time.sleep(5 * (attempt + 1))
    record(uid, status="failed", error="See prior retry records and asset_download.log")


if __name__ == "__main__":
    # Probe once on each invocation, so a recovered upstream host is retried.
    try:
        with requests.get(DATA_SOURCES["pick_clutter_ycb_configs"].url,
                          stream=True, timeout=(5, 10)) as probe:
            UCSD_AVAILABLE = probe.status_code == 200
    except requests.RequestException:
        pass
    # ReplicaCADRearrange is nested inside ReplicaCAD; install it after the parent.
    priority = ["ycb", "pick_clutter_ycb_configs", "bridge_v2_real2sim"]
    uids = priority + [u for u in DATA_SOURCES if u not in priority and u != "ReplicaCADRearrange"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run_asset, uids):
            pass
    run_asset("ReplicaCADRearrange")
    failures = [u for u in DATA_SOURCES if state["assets"].get(u, {}).get("status") != "complete"]
    state["finished_at"] = time.time()
    record("_campaign", status="complete" if not failures else "incomplete", failed_assets=failures)
    raise SystemExit(bool(failures))
