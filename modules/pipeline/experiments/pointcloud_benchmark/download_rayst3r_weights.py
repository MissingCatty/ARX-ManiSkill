"""下载并核验固定版本的 RaySt3R 公开权重。"""
import hashlib
import concurrent.futures
import json
import os
from pathlib import Path
import threading
import time
import requests

os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "weights" / "rayst3r"
REPO = "bartduis/rayst3r"
REVISION = "f0adb75ae9df78ef80b0bdf5b05c41eb27b89923"
NAME = "rayst3r.pth"
STATE = threading.local()


def metadata(record):
    urls = [f"https://huggingface.co/api/models/{REPO}", f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}"]
    session = requests.Session()
    for attempt in range(8):
        url = urls[attempt % len(urls)]
        record.update(status="fetching_metadata", metadata_attempt=attempt + 1)
        save(record)
        try:
            response = session.get(url, params={"blobs": "true"}, timeout=(15, 25))
            response.raise_for_status()
            info = response.json()
            if info.get("sha") != REVISION:
                raise RuntimeError("RaySt3R metadata revision mismatch")
            spec = next(f for f in info["siblings"] if f["rfilename"] == NAME)
            expected = spec["lfs"]["sha256"]
            if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise RuntimeError("Invalid RaySt3R upstream checksum")
            size = spec.get("size", spec["lfs"]["size"])
            (DEST / "upstream_metadata.json").write_text(json.dumps({"url": url, "revision": info["sha"], "file": spec}, ensure_ascii=False, indent=2) + "\n")
            return expected, size
        except (requests.RequestException, ValueError, RuntimeError, KeyError, StopIteration) as exc:
            record.update(last_metadata_error=f"{type(exc).__name__}: {exc}")
            save(record)
            print(f"RaySt3R 元数据连接重试 {attempt + 1}/8: {type(exc).__name__}", flush=True)
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 15))


def download(expected_size):
    part_dir = DEST / "parts"
    part_dir.mkdir(exist_ok=True)
    chunk_size = 8 * 1024 * 1024
    count = (expected_size + chunk_size - 1) // chunk_size

    def fetch(index):
        if not hasattr(STATE, "session"):
            STATE.session = requests.Session()
        start, end = index * chunk_size, min((index + 1) * chunk_size, expected_size) - 1
        part = part_dir / f"{index:06d}"
        if part.exists() and part.stat().st_size == end - start + 1:
            return
        for attempt in range(8):
            offset = part.stat().st_size if part.exists() else 0
            if offset == end - start + 1:
                return
            try:
                url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{NAME}"
                first = start + offset
                with STATE.session.get(url, params={"download": "true", "range_start": first}, headers={"Range": f"bytes={first}-{end}"}, timeout=(30, 120), stream=True) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {first}-{end}/{expected_size}":
                        raise RuntimeError("Unexpected HF byte-range response")
                    with part.open("ab") as f:
                        for data in response.iter_content(64 * 1024):
                            f.write(data)
                if part.stat().st_size != end - start + 1:
                    raise RuntimeError("Truncated HF range response")
                return
            except (requests.RequestException, RuntimeError):
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 8))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index, _ in enumerate(pool.map(fetch, range(count)), 1):
            if index % 16 == 0 or index == count:
                print(f"RaySt3R: {index}/{count} 分块", flush=True)
    temp = DEST / (NAME + ".partial")
    with temp.open("wb") as f:
        for index in range(count):
            with (part_dir / f"{index:06d}").open("rb") as src:
                for data in iter(lambda: src.read(8 * 1024 * 1024), b""):
                    f.write(data)
    return temp, part_dir


def save(data):
    path = DEST / "download_receipt.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    record = {"repository": REPO, "revision": REVISION, "status": "fetching_metadata", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    save(record)
    try:
        expected, size = metadata(record)
        record.update(status="downloading", expected_sha256=expected, expected_size=size)
        save(record)
        target = DEST / NAME
        if target.exists() and target.stat().st_size == size:
            h = hashlib.sha256()
            with target.open("rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() == expected:
                record.update(status="complete", verified_sha256=expected, reused_existing=True)
                save(record)
                return
        path, parts = download(size)
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        if path.stat().st_size != size or h.hexdigest() != expected:
            raise RuntimeError("RaySt3R weight integrity mismatch")
        path.replace(DEST / NAME)
        for part in parts.iterdir():
            part.unlink()
        parts.rmdir()
        record.update(status="complete", verified_sha256=h.hexdigest(), completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        save(record)
        print("RaySt3R 下载与SHA256校验完成", flush=True)
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(record)
        raise


if __name__ == "__main__":
    main()
