"""
src/utils/logger.py
Logger tập trung cho toàn bộ hệ thống CWS.
Ghi log ra console + file CSV + file JSON theo session.
"""
import csv
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Màu terminal ──────────────────────────────────────────
_RESET  = "\033[0m"
_COLORS = {
    "DEBUG"   : "\033[37m",
    "INFO"    : "\033[36m",
    "WARNING" : "\033[33m",
    "ERROR"   : "\033[31m",
    "CRITICAL": "\033[35m",
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = _COLORS.get(record.levelname, _RESET)
        record.levelname = f"{color}{record.levelname:<8}{_RESET}"
        return super().format(record)


def get_logger(name: str = "CWS", level: int = logging.INFO) -> logging.Logger:
    """Lấy logger đã cấu hình với màu sắc cho console."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(ColorFormatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                    datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    logger.propagate = False
    return logger


# ── CSV Event Logger ──────────────────────────────────────
class EventLogger:
    """
    Ghi log sự kiện cảnh báo ra CSV.
    Mỗi hàng: timestamp, alert_level, distance_m, ttc_s, class_name, track_id, speed
    """
    HEADER = ["timestamp", "alert_level", "distance_m",
              "ttc_s", "class_name", "track_id", "approach_speed_ms"]

    def __init__(self, log_path: str = "logs/events.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_start = datetime.now().isoformat()
        self._counts        = {"SAFE": 0, "WARNING": 0, "DANGER": 0}

        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADER)

    def log(
        self,
        alert_level   : str,
        distance_m    : float,
        ttc_s         : Optional[float] = None,
        class_name    : str = "",
        track_id      : int = -1,
        approach_speed: float = 0.0,
    ):
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row = [
            ts,
            alert_level,
            f"{distance_m:.2f}",
            f"{ttc_s:.2f}" if ttc_s is not None else "",
            class_name,
            track_id,
            f"{approach_speed:.3f}",
        ]
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

        level_str = alert_level if isinstance(alert_level, str) else alert_level.value
        self._counts[level_str] = self._counts.get(level_str, 0) + 1

    def save_session_summary(self, output_path: str = "logs/session_stats.json",
                              extra: dict = None):
        summary = {
            "session_start": self._session_start,
            "session_end"  : datetime.now().isoformat(),
            "event_counts" : self._counts,
            "log_file"     : str(self.log_path),
        }
        if extra:
            summary.update(extra)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    def get_stats(self) -> dict:
        return self._counts.copy()


# ── FPS Counter ───────────────────────────────────────────
class FPSCounter:
    """Đo FPS trung bình trên cửa sổ trượt N frame."""

    def __init__(self, window: int = 30):
        self._times : list = []
        self._window = window

    def tick(self):
        """Gọi mỗi frame."""
        self._times.append(time.perf_counter())
        if len(self._times) > self._window + 1:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def ms(self) -> float:
        return 1000.0 / self.fps if self.fps > 0 else 0.0
