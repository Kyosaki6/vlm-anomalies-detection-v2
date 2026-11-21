# -*- coding: utf-8 -*-
"""Một cửa sổ cho smoke P2: k khung cách đều trên toàn bộ video.

Tách khỏi `scripts/qwen3_smoke.py` để file đó không import cv2. Máy không có GPU đọc gate
và thoát trước; chỉ máy chạy smoke mới cần OpenCV.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vad.io.video import probe_video


def load_window(video: Path, k: int) -> tuple[list[np.ndarray], dict[str, str | float]]:
    """-> (k khung RGB, context cho `VLMVerifier.verify`). Video hỏng raise."""
    if not video.is_file():
        raise FileNotFoundError(f"không thấy video: {video}")
    # Import muộn: run_verify kéo theo cv2 ngay khi import, còn smoke đọc gate trước.
    from scripts.run_verify import frame_indices_for_times, pick_frame_times, read_frames_at

    n_frames, fps = probe_video(video)
    times = pick_frame_times(0.0, n_frames / fps, k)
    indices = frame_indices_for_times(times, fps, n_frames)
    frames_by_index = read_frames_at(video, indices)
    frames = [frames_by_index[i] for i in indices]
    context: dict[str, str | float] = {
        "video_id": video.stem,
        "start_s": 0.0,
        "end_s": n_frames / fps,
    }
    return frames, context
