"""从 ModelScope 下载公开 SPAR3D 镜像，断点续传并验证每个文件的 SHA256。"""
from __future__ import annotations

import concurrent.futures
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import requests

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "weights" / "spar3d"
REPO = "stabilityai/stable-point-aware-3d"
API = f"https://modelscope.cn/api/v1/models/{REPO}"
CHUNK = 32 * 1024 * 1024
FILES = ("config.yaml", "LICENSE.md", "README.md", "model.safetensors")
THREAD_STATE = threading.local()


def session():
    if not hasattr(THREAD_STATE, "session"):
        THREAD_STATE.session = requests.Session()
        # ModelScope is reachable directly from this host. Avoid routing its
        # large files through the environment's external-site proxy.
        THREAD_STATE.session.trust_env = False
    return THREAD_STATE.session


def get_json(url, **kwargs):
    for attempt in range(5):
        try:
            response = session().get(url, timeout=(15, 60), **kwargs)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    receipt_path = DEST / "download_receipt.json"
    info = get_json(API)["Data"]
    listing = get_json(API + "/repo/files", params={"Revision": "master", "Recursive": "true"})["Data"]
    entries = {f["Path"]: f for f in listing["Files"]}
    missing = [name for name in FILES if name not in entries]
    if missing:
        raise RuntimeError(f"Files absent from repository: {missing}")
    save_json(DEST / "repository_files.json", listing)
    receipt = {
        "repository": REPO,
        "url": f"https://modelscope.cn/models/{REPO}",
        "provider": "ModelScope",
        "organization_display_name": info.get("Organization", {}).get("FullName"),
        "model_source": info.get("ModelSource"),
        "commit": listing["LatestCommitter"].get("Id") or listing["LatestCommitter"].get("ShortId"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "downloading",
        "files": {},
    }
    save_json(receipt_path, receipt)
    for name in FILES:
        entry = entries[name]
        size, expected = entry["Size"], entry["Sha256"]
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        detail = {"size": size, "sha256": expected, "file_revision": entry["Revision"], "status": "downloading"}
        receipt["files"][name] = detail
        save_json(receipt_path, receipt)
        if target.exists() and target.stat().st_size == size and digest(target) == expected:
            detail["status"] = "verified_existing"
            save_json(receipt_path, receipt)
            print(f"已核验 {name}", flush=True)
            continue
        parts = DEST / (name + ".parts")
        parts.mkdir(exist_ok=True)
        count = (size + CHUNK - 1) // CHUNK

        def fetch(index: int) -> Path:
            start = index * CHUNK
            end = min(size, start + CHUNK) - 1
            part = parts / f"{index:06d}"
            if part.exists() and part.stat().st_size == end - start + 1:
                return part
            # Include the byte range in the query to avoid reusing cached responses
            # for a different Range header on the same URL.
            url = f"https://modelscope.cn/models/{REPO}/resolve/{entry['Revision']}/{name}"
            for attempt in range(5):
                try:
                    with session().get(url, params={"download": "true", "range_start": start}, headers={"Range": f"bytes={start}-{end}"}, timeout=(30, 120), stream=True) as response:
                        response.raise_for_status()
                        if response.status_code not in (200, 206) or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                            raise RuntimeError(f"Unexpected range response: {response.status_code}, {response.headers.get('Content-Range')}")
                        temp = part.with_suffix(".partial")
                        with temp.open("wb") as out:
                            for data in response.iter_content(1024 * 1024):
                                out.write(data)
                        if temp.stat().st_size != end - start + 1:
                            raise RuntimeError("Truncated download")
                        temp.replace(part)
                    return part
                except (requests.RequestException, OSError, RuntimeError):
                    if attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 8))
            raise AssertionError("unreachable")

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for _ in pool.map(fetch, range(count)):
                completed += 1
                if completed == count or completed % 8 == 0:
                    detail["completed_chunks"] = completed
                    detail["total_chunks"] = count
                    save_json(receipt_path, receipt)
                    print(f"{name}: {completed}/{count} 分块", flush=True)
        temp_target = target.with_suffix(target.suffix + ".partial")
        h = hashlib.sha256()
        with temp_target.open("wb") as out:
            for index in range(count):
                with (parts / f"{index:06d}").open("rb") as f:
                    for data in iter(lambda: f.read(8 * 1024 * 1024), b""):
                        out.write(data)
                        h.update(data)
        actual = h.hexdigest()
        if temp_target.stat().st_size != size or actual != expected:
            raise RuntimeError(f"Integrity failure: {name}, SHA256={actual}")
        temp_target.replace(target)
        for part in parts.iterdir():
            part.unlink()
        parts.rmdir()
        detail["status"] = "verified"
        detail["verified_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save_json(receipt_path, receipt)
        print(f"下载并核验完成 {name}: {size} bytes, SHA256={actual}", flush=True)
    receipt["status"] = "complete"
    receipt["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save_json(receipt_path, receipt)
    print(f"{REPO} 镜像权重全部下载并核验完成。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--directory", default="spar3d")
    parser.add_argument("--files", nargs="+", default=FILES)
    args = parser.parse_args()
    REPO = args.repo
    DEST = ROOT / "weights" / args.directory
    API = f"https://modelscope.cn/api/v1/models/{REPO}"
    FILES = tuple(args.files)
    try:
        main()
    except Exception as exc:
        path = DEST / "download_receipt.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            receipt["status"] = "failed"
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            receipt["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            save_json(path, receipt)
        raise
