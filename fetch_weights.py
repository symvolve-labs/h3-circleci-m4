"""Fetch pinned model files on the benchmark VM. Select a mirror before the timed trials."""
from h3_image_mlx.models import MANIFEST
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import httpx
from h3_image_mlx.download import _download_file
import subprocess
import time

root = Path("weights").resolve()
root.mkdir(exist_ok=True)
entries = MANIFEST["files"]
if len(sys.argv) == 2:
    # Reuse the pinned release's range, length and SHA-256 verification.
    mirror = sys.argv[1]
    assert mirror in ("modelscope", "huggingface")
    from h3_image_mlx.models import sha256

    def fetch_entry(entry):
        dest = root / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.stat().st_size != entry["bytes"] or sha256(dest) != entry["sha256"]:
                raise ValueError(f"Invalid completed file: {dest}")
            return
        with httpx.Client(follow_redirects=True, timeout=45) as client:
            _download_file(client, entry[mirror], dest, entry)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch_entry, entries))
    raise SystemExit(0)

total = sum(entry["bytes"] for entry in entries)
largest = max(entries, key=lambda e: e["bytes"])
rates = {}
probe_size = 8 * 1024 * 1024
for mirror in ("modelscope", "huggingface"):
    started = time.monotonic()
    probe = subprocess.run(
        ["curl", "-sSfL", "--max-time", "15", "--range", f"0-{probe_size - 1}",
         "--max-filesize", str(probe_size), "-o", "/dev/null", "-w",
         "%{http_code} %{size_download}", largest[mirror]],
        capture_output=True, text=True)
    if probe.returncode == 0 and probe.stdout.strip() == f"206 {probe_size}":
        rates[mirror] = probe_size / (time.monotonic() - started)
print("Sampled mirror MB/s:", {k: round(v / 1e6, 2) for k, v in rates.items()}, flush=True)
mirrors = sorted(("huggingface", "modelscope"), key=lambda k: rates.get(k, 0), reverse=True)
if rates.get("modelscope", 0) >= 30e6:
    mirrors = ["modelscope", "huggingface"]

def downloaded():
    return sum(path.stat().st_size for e in entries
               for path in (root / e["path"], root / (e["path"] + ".part"))
               if path.is_file())

deadline = time.monotonic() + 20 * 60
for mirror in mirrors:
    previous = downloaded()
    start = time.monotonic()
    print("Downloading from", mirror, flush=True)
    with subprocess.Popen([sys.executable, __file__, mirror]) as proc:
        while time.monotonic() < deadline:
            try:
                status = proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start
                size = downloaded()
                speed = max(0, size - previous) / elapsed
                print(f"Transfer {size / 1e9:.2f}/{total / 1e9:.2f} GB, {speed / 1e6:.1f} MB/s", flush=True)
                alternate_faster = rates.get(mirrors[-1], 0) > speed or size == previous
                if (mirror == mirrors[0] and elapsed >= 90 and size < total and alternate_faster
                        and (total - size) / max(1, speed) > deadline - time.monotonic()):
                    print("Switching mirrors to fit the benchmark download window.", flush=True)
                    break
                continue
            if status == 0:
                print("Release manifest hashes verified.", flush=True)
                raise SystemExit(0)
            break
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    if time.monotonic() >= deadline:
        break
raise SystemExit("Could not verify all pinned weights within the download window.")
