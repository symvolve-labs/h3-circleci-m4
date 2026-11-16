#!/bin/bash
set -euo pipefail

ROOT=$(pwd)
test "$(/usr/bin/uname -m)" = arm64
python3 - <<'PY'
import platform, re, subprocess
cpu = subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"], text=True).strip()
system = platform.mac_ver()[0]
print("Guest CPU:", cpu, "macOS:", system, flush=True)
if not re.fullmatch(r"Apple M4 Pro(?: \(Virtual\))?", cpu) or tuple(map(int, system.split(".")[:2])) >= (26, 4):
    raise SystemExit("Requires an M4 Pro running macOS earlier than 26.4")
PY

# Fetch the public runtime and measurement code, with no model files.
SOURCE=https://github.com/Anionex/h3-image-mlx.git
pin() {
  git init -q "$1"
  git -C "$1" fetch -q --depth 1 "$SOURCE" "$2"
  git -C "$1" checkout -q --detach FETCH_HEAD
  test "$(git -C "$1" rev-parse HEAD)" = "$2"
}
pin candidate c5b3f8300a5876356806a2adef815c18b2476395
pin harness f502af0594821f4a74bc05c3d13dcd68a192ade9
if ! command -v uv >/dev/null; then
  python3 -m venv "$ROOT/bootstrap"
  "$ROOT/bootstrap/bin/python" -m pip install 'uv==0.8.22'
  export PATH="$ROOT/bootstrap/bin:$PATH"
fi
uv venv --python 3.13 "$ROOT/runtime"
export PATH="$ROOT/runtime/bin:$PATH"
uv pip install --python "$ROOT/runtime/bin/python" 'mlx==0.32.4' 'h3-image-mlx==0.1.0a8'
mkdir -p evidence
PYTHONPATH="$ROOT/candidate/src:$ROOT/harness/scripts/legacy_bench" python - <<'PY' | tee evidence/probe.log
import subprocess
import mlx.core as mx
from unittest.mock import patch
from h3_image_mlx import _portable
from cache_stats import disk_bytes
print("Metal device:", mx.device_info(), flush=True)
assert mx.metal.is_available() and _portable._legacy_metal()
mx.set_default_device(mx.gpu)
assert disk_bytes() > 0, "Guest block-device counters unavailable"
rows, width, channels = 513, 1088, 64
with patch.object(_portable, "_large_portable_matmul", wraps=_portable._large_portable_matmul) as dispatch:
    output = _portable.portable_matmul(mx.full((rows,width), -128, dtype=mx.int8),
        mx.full((channels,width), 127, dtype=mx.int8), mx.ones((rows,)), mx.ones((channels,)), mx.float32)
    assert mx.all(output == -128*127*width).item()
    dispatch.assert_called_once()
print("H3 large Metal SIMD output matched the integer reference.", flush=True)
PY
PYTHONPATH="$ROOT/candidate/src:$ROOT/candidate/tests" python -m unittest \
  test_portable.PortableProjectionTests.test_simd_reads_byte_aligned_dlpack_weight \
  test_portable.PortableProjectionTests.test_legacy_large_simd_projection_extremes_and_partial_tiles
python - <<'PY'
import json, shutil
from pathlib import Path
manifest = json.loads(Path("candidate/src/h3_image_mlx/weights.json").read_text())
required = sum(item["bytes"] for item in manifest["files"]) + 2 * 1024**3
free = shutil.disk_usage(".").free
print("Free and required disk bytes:", free, required, flush=True)
if free < required:
    raise SystemExit("Insufficient disk space for full model.")
PY
if ! h3-image-mlx fetch --models "$ROOT/weights" --source modelscope; then
  h3-image-mlx fetch --models "$ROOT/weights" --source huggingface
fi
sudo -n purge
python harness/scripts/legacy_bench/compare.py --candidate candidate --weights weights --results results --pairs 3
