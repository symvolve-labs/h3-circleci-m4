# H3 on CircleCI M4 Pro

This manual job compares H3 Image MLX 0.1.0a8 with the legacy SIMD candidate on one M4 Pro, using CircleCI's Xcode 26.4.1 image (macOS 26.3). Set the boolean pipeline parameter `run_benchmark` to true after verifying the organization's free plan and available credits.

The job checks the guest chip, macOS version, actual H3 GPU dispatch, integer output and available disk before fetching the pinned weights. It renders six 1024×1024 images in interleaved order with the same seed and six denoising steps. The results include PNG hashes, end-to-end time, peak stage-process footprint, file residency and guest disk reads. Only the measurement directory is uploaded as an artifact.
