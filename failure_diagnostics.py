"""Record intermediate finiteness only after an uncaught stage exception."""
import sys

previous = sys.excepthook

def record_stage_failure(kind, exc, trace):
    previous(kind, exc, trace)
    if len(sys.argv) != 3 or not sys.argv[0].endswith("/_stage.py"):
        return
    from pathlib import Path
    import json
    import os
    import mlx.core as mx

    work = Path(sys.argv[2])
    record = {"stage": sys.argv[1], "error": str(exc), "tensors": {}}
    for name in ("embedding", "latent"):
        source = work / (name + ".safetensors")
        if source.is_file():
            with mx.stream(mx.cpu):
                for key, array in mx.load(str(source)).items():
                    record["tensors"][key] = {
                        "shape": list(array.shape),
                        "finite": bool(mx.all(mx.isfinite(array.astype(mx.float32))).item()),
                    }
    (Path(os.environ["H3_TRIAL_DIR"]) / "stage-failure.json").write_text(json.dumps(record, indent=2))

sys.excepthook = record_stage_failure
