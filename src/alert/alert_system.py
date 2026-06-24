"""
=============================================================
MODULE: He thong canh bao & Am thanh (TTS)
src/alert/alert_system.py
=============================================================
Tinh nang:
  - 3 muc canh bao: SAFE / WARNING / DANGER
  - Am thanh TTS tieng Viet: "Vat can cach ban 3 met, con 2 giay va cham!"
  - Cooldown timer tranh phat lien tuc
  - Cache file am thanh MP3 de phat offline
  - Fallback: beep am thanh don gian khi khong co mang

Cach dung:
  python src/alert/alert_system.py --demo
  python src/alert/alert_system.py --test-tts
  python src/alert/alert_system.py --calibrate-thresholds
"""

import os
import sys
import time
import threading
import queue
import json
import hashlib
import tempfile
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path


# ── Cau hinh nguong canh bao ──────────────────────────────
class AlertLevel(Enum):
    SAFE    = "SAFE"
    WARNING = "WARNING"
    DANGER  = "DANGER"


ALERT_CONFIG = {
    # Nguong TTC (giay)
    "ttc_warning"  : 5.0,
    "ttc_danger"   : 2.0,
    # Nguong khoang cach (met)
    "dist_warning" : 8.0,
    "dist_danger"  : 3.0,
    # Cooldown giua cac canh bao am thanh (giay)
    "cooldown_safe"   : 10.0,
    "cooldown_warning":  3.0,
    "cooldown_danger" :  1.5,
    # Thu muc cache am thanh
    "audio_cache_dir" : "data/audio_cache",
    # Ngon ngu TTS
    "tts_lang"        : "vi",
}

# Mau sac BGR cho overlay
ALERT_COLORS = {
    AlertLevel.SAFE    : (50,  200, 50),
    AlertLevel.WARNING : (0,   165, 255),
    AlertLevel.DANGER  : (0,   0,   230),
}

# Mau background panel
ALERT_BG = {
    AlertLevel.SAFE    : (20,  80,  20),
    AlertLevel.WARNING : (20,  60, 120),
    AlertLevel.DANGER  : (40,   0,   0),
}

# Nhan hien thi
ALERT_LABELS_VI = {
    AlertLevel.SAFE    : "AN TOAN",
    AlertLevel.WARNING : "CANH BAO",
    AlertLevel.DANGER  : "NGUY HIEM !!!",
}


# ══════════════════════════════════════════════════════════
#  ENGINE AM THANH
# ══════════════════════════════════════════════════════════
class AudioEngine:
    """
    Engine phat am thanh TTS + beep.
    Thu tu uu tien:
      1. gTTS + pygame (chat luong cao, can mang lan dau)
      2. pyttsx3 (offline, chat luong thap hon)
      3. Beep system (fallback cuoi cung)
    """

    def __init__(self, lang: str = "vi", cache_dir: str = "data/audio_cache"):
        self.lang       = lang
        self.cache_dir  = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._engine_type = self._detect_engine()
        self._pyttsx3_engine = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        if self._engine_type == "gtts+pygame":
            self._init_pygame()
        elif self._engine_type == "pyttsx3":
            self._init_pyttsx3()

        print(f"[AudioEngine] Su dung engine: {self._engine_type}")

    def stop(self):
        self._stop_event.set()
        try:
            if self._engine_type == "gtts+pygame":
                import pygame
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
                    pygame.mixer.stop()  # Ngat toan bo tieng beep (sound channels)
            elif self._engine_type == "pyttsx3":
                if self._pyttsx3_engine:
                    self._pyttsx3_engine.stop()
        except Exception:
            pass

    def _detect_engine(self) -> str:
        try:
            import pygame
            import gtts
            return "gtts+pygame"
        except ImportError:
            pass
        try:
            import pyttsx3
            return "pyttsx3"
        except ImportError:
            pass
        return "beep"

    def _init_pygame(self):
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        except Exception as e:
            print(f"  [AudioEngine] pygame init error: {e}")
            self._engine_type = "beep"

    def _init_pyttsx3(self):
        try:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            # Tim giong Viet neu co
            voices = self._pyttsx3_engine.getProperty("voices")
            for v in voices:
                if "vi" in v.id.lower() or "viet" in v.name.lower():
                    self._pyttsx3_engine.setProperty("voice", v.id)
                    break
            self._pyttsx3_engine.setProperty("rate", 160)   # toc do doc
            self._pyttsx3_engine.setProperty("volume", 0.9)
        except Exception as e:
            print(f"  [AudioEngine] pyttsx3 init error: {e}")
            self._engine_type = "beep"

    # ── Lay file cache (hoac tao moi) ─────────────────────
    def _get_cached_audio(self, text: str) -> Optional[Path]:
        """Lay file MP3 tu cache. Neu chua co, tao bang gTTS."""
        key      = hashlib.md5(f"{self.lang}:{text}".encode()).hexdigest()[:12]
        mp3_path = self.cache_dir / f"{key}.mp3"

        # Kiem tra neu file da ton tai va khong bi loi (dung luong > 0 bytes)
        if mp3_path.exists() and mp3_path.stat().st_size > 0:
            return mp3_path
        elif mp3_path.exists():
            # Xoa file loi 0 bytes neu co
            try:
                mp3_path.unlink()
            except Exception:
                pass

        # Tao moi bang gTTS
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=self.lang, slow=False)
            tts.save(str(mp3_path))
            return mp3_path
        except Exception as e:
            print(f"  [AudioEngine] gTTS error: {e}")
            return None

    # ── Phat am thanh (non-blocking) ──────────────────────
    def speak(self, text: str, block: bool = False):
        """Phat TTS. Non-blocking theo mac dinh."""
        if self._engine_type == "gtts+pygame":
            t = threading.Thread(target=self._speak_gtts, args=(text,), daemon=True)
            t.start()
            if block:
                t.join()
        elif self._engine_type == "pyttsx3":
            t = threading.Thread(target=self._speak_pyttsx3, args=(text,), daemon=True)
            t.start()
            if block:
                t.join()
        else:
            self._beep_fallback(text)

    def _speak_gtts(self, text: str):
        try:
            with self._lock:
                if self._stop_event.is_set(): return
                mp3_path = self._get_cached_audio(text)
                if not mp3_path or not mp3_path.exists() or mp3_path.stat().st_size == 0:
                    self._beep_fallback(text)
                    return
                try:
                    import pygame
                    if not pygame.mixer.get_init():
                        self._init_pygame()
                    pygame.mixer.music.load(str(mp3_path))
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy() and not self._stop_event.is_set():
                        time.sleep(0.05)
                except Exception as e:
                    print(f"  [AudioEngine] play error: {e}. Dang thu khoi tao lai mixer...")
                    # Thu re-init pygame neu bi loi thiet bi am thanh chua mo
                    try:
                        import pygame
                        pygame.mixer.quit()
                        self._init_pygame()
                        pygame.mixer.music.load(str(mp3_path))
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() and not self._stop_event.is_set():
                            time.sleep(0.05)
                    except Exception as re_err:
                        print(f"  [AudioEngine] play retry error: {re_err}")
                        self._beep_fallback(text)
        except (TypeError, AttributeError, Exception):
            pass

    def _speak_pyttsx3(self, text: str):
        try:
            with self._lock:
                if self._stop_event.is_set(): return
                try:
                    self._pyttsx3_engine.say(text)
                    self._pyttsx3_engine.runAndWait()
                except Exception as e:
                    print(f"  [AudioEngine] pyttsx3 speak error: {e}")
        except (TypeError, AttributeError, Exception):
            pass

    def _beep_fallback(self, text: str):
        """Phat beep ASCII khi khong co engine TTS."""
        print(f"\a  [ALERT BEEP] {text}")   # \a = ASCII bell

    def beep(self, freq_hz: int = 880, duration_ms: int = 200):
        """Phat tieng beep ngan."""
        try:
            with self._lock:
                if self._stop_event.is_set(): return
                import pygame
                if not pygame.mixer.get_init():
                    self._init_pygame()
                sample_rate = 22050
                n_samples   = int(sample_rate * duration_ms / 1000)
                import numpy as np
                t   = np.linspace(0, duration_ms/1000, n_samples, False)
                wav = (np.sin(2 * np.pi * freq_hz * t) * 28000).astype(np.int16)
                sound = pygame.sndarray.make_sound(wav.reshape(-1, 1).repeat(2, axis=1))
                sound.play()
                time.sleep(duration_ms / 1000 + 0.05)
        except Exception:
            print("\a")

    def pre_cache_messages(self, messages: List[str]):
        """Pre-generate va cache cac cau canh bao thuong dung."""
        if self._engine_type != "gtts+pygame":
            return
        print("[AudioEngine] Dang cache am thanh...")
        for msg in messages:
            path = self._get_cached_audio(msg)
            status = "OK" if path else "FAIL"
            print(f"  [{status}] {msg}")


# ══════════════════════════════════════════════════════════
#  CLASS CHINH: AlertSystem
# ══════════════════════════════════════════════════════════
class AlertSystem:
    """
    He thong canh bao ket hop:
      - Xac dinh muc canh bao tu TTC + khoang cach
      - Phat am thanh TTS tieng Viet
      - Cooldown timer tranh spam
      - Queue am thanh xu ly bat dong bo
    """

    def __init__(self, config: dict = None):
        self.cfg    = config or ALERT_CONFIG.copy()
        self.audio  = AudioEngine(
            lang      = self.cfg["tts_lang"],
            cache_dir = self.cfg["audio_cache_dir"],
        )
        # Thoi diem phat canh bao cuoi cung
        self._last_alert_time : Dict[AlertLevel, float] = {
            lvl: 0.0 for lvl in AlertLevel
        }
        self._current_level   = AlertLevel.SAFE
        self._alert_queue     = queue.Queue()
        self._stats           = {"safe": 0, "warning": 0, "danger": 0, "total_alerts": 0}
        self._stop_event      = threading.Event()

        # Pre-cache cac cau hay dung
        self._precache_common_messages()

    def stop(self):
        self._stop_event.set()
        self.audio.stop()

    # ── Pre-cache ─────────────────────────────────────────
    def _precache_common_messages(self):
        msgs = self._generate_distance_messages()
        threading.Thread(
            target=self.audio.pre_cache_messages,
            args=(msgs[:20],),
            daemon=True
        ).start()

    def _generate_distance_messages(self) -> List[str]:
        """Sinh cac cau canh bao cho cac khoang cach pho bien."""
        messages = []
        for d in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
            for t in [1, 2, 3, 4, 5]:
                messages.append(
                    f"Canh bao! Vat can cach ban {d} met, con {t} giay co the va cham!"
                )
            messages.append(f"Nguy hiem! Vat can chi con {d} met!")
        messages += [
            "Canh bao! Co vat can phia truoc!",
            "Nguy hiem! Phao truoc qua gan!",
            "Giu khoang cach an toan!",
            "Canh bao toc do!",
        ]
        return messages

    # ── Xac dinh muc canh bao ─────────────────────────────
    def evaluate(
        self,
        distance_m : float,
        ttc_s      : Optional[float] = None,
        class_name : str = "obstacle",
    ) -> AlertLevel:
        """
        Xac dinh AlertLevel tu khoang cach va TTC.
        Logic: uu tien khoang cach tuyet doi truoc, sau do TTC.
        """
        # Nguong khoang cach tuyet doi
        if distance_m <= self.cfg["dist_danger"]:
            return AlertLevel.DANGER
        if distance_m <= self.cfg["dist_warning"]:
            return AlertLevel.WARNING

        # Nguong TTC
        if ttc_s is not None and ttc_s > 0:
            if ttc_s <= self.cfg["ttc_danger"]:
                return AlertLevel.DANGER
            if ttc_s <= self.cfg["ttc_warning"]:
                return AlertLevel.WARNING

        return AlertLevel.SAFE

    # ── Tao noi dung canh bao ─────────────────────────────
    def _build_message(
        self,
        level      : AlertLevel,
        distance_m : float,
        ttc_s      : Optional[float],
        class_name : str,
    ) -> str:
        """Tao cau canh bao tieng Viet tu nhien."""
        obj_name = {
            "car"       : "xe o to",
            "motorbike" : "xe may",
            "person"    : "nguoi di bo",
            "truck"     : "xe tai",
            "obstacle"  : "vat can",
        }.get(class_name, "vat can")

        d_str = f"{distance_m:.0f}" if distance_m >= 1 else f"{distance_m:.1f}"

        if level == AlertLevel.DANGER:
            if ttc_s is not None and 0 < ttc_s < 30:
                return f"Nguy hiem! {obj_name} chi con {d_str} met, con {ttc_s:.0f} giay co the va cham!"
            else:
                return f"Nguy hiem! {obj_name} qua gan, chi con {d_str} met!"
        elif level == AlertLevel.WARNING:
            if ttc_s is not None and 0 < ttc_s < 30:
                return f"Canh bao! {obj_name} cach ban {d_str} met, con {ttc_s:.0f} giay!"
            else:
                return f"Canh bao! {obj_name} cach ban {d_str} met, giu khoang cach!"
        return ""

    # ── Kich hoat canh bao ────────────────────────────────
    def trigger(
        self,
        distance_m : float,
        ttc_s      : Optional[float] = None,
        class_name : str = "obstacle",
        force      : bool = False,
    ) -> AlertLevel:
        """
        Kiem tra va kich hoat canh bao neu can.

        Args:
            distance_m : Khoang cach den vat the (met)
            ttc_s      : Time-to-collision (giay), None neu khong tinh duoc
            class_name : Ten class cua vat the
            force      : Phat canh bao bat ke cooldown

        Returns:
            AlertLevel hien tai
        """
        level = self.evaluate(distance_m, ttc_s, class_name)
        self._current_level = level

        # Dem thong ke
        self._stats[level.value.lower()] += 1

        if level == AlertLevel.SAFE:
            return level

        # Kiem tra cooldown
        now      = time.time()
        cooldown = self.cfg[f"cooldown_{level.value.lower()}"]
        last     = self._last_alert_time[level]

        if not force and (now - last) < cooldown:
            return level   # Con trong cooldown, khong phat

        # Phat canh bao
        self._last_alert_time[level] = now
        self._stats["total_alerts"] += 1

        msg = self._build_message(level, distance_m, ttc_s, class_name)

        if level == AlertLevel.DANGER:
            # Phat beep khan cap truoc, roi TTS
            threading.Thread(target=self._play_danger_sequence, args=(msg,), daemon=True).start()
        else:
            # WARNING: chi TTS
            if msg:
                self.audio.speak(msg)

        return level

    def _play_danger_sequence(self, msg: str):
        """Beep x2 + TTS cho DANGER."""
        if self._stop_event.is_set(): return
        self.audio.beep(freq_hz=1200, duration_ms=150)
        if self._stop_event.is_set(): return
        time.sleep(0.1)
        if self._stop_event.is_set(): return
        self.audio.beep(freq_hz=1200, duration_ms=150)
        if self._stop_event.is_set(): return
        time.sleep(0.1)
        if self._stop_event.is_set(): return
        if msg:
            self.audio.speak(msg)

    # ── Trigger tu track output (Ngay 6) ──────────────────
    def process_tracks(self, tracks_output: List[dict]) -> AlertLevel:
        """
        Nhan danh sach track tu ObjectTracker (Ngay 6) va xu ly canh bao.
        tracks_output: output cua ObjectTracker.update()

        Returns:
            Muc canh bao cao nhat hien tai
        """
        if not tracks_output:
            return AlertLevel.SAFE

        # Tim vat nguy hiem nhat (da duoc sort: DANGER truoc)
        worst = tracks_output[0]
        level = self.trigger(
            distance_m = worst["distance_m"],
            ttc_s      = worst.get("ttc"),
            class_name = worst.get("class_name", "obstacle"),
        )
        return level

    @property
    def current_level(self) -> AlertLevel:
        return self._current_level

    def get_stats(self) -> dict:
        return self._stats.copy()

    def reset_cooldowns(self):
        for lvl in AlertLevel:
            self._last_alert_time[lvl] = 0.0


# ══════════════════════════════════════════════════════════
#  OVERLAY RENDERER
# ══════════════════════════════════════════════════════════
class AlertOverlayRenderer:
    """
    Ve toan bo thong tin canh bao len frame OpenCV.
    Tach rieng viec ve (UI) khoi logic canh bao.
    """

    def __init__(self):
        self._flash_state = True
        self._last_flash  = 0.0
        self._flash_interval = 0.4   # giay

    def render(
        self,
        frame         : 'np.ndarray',
        tracks_output : List[dict],
        alert_level   : AlertLevel,
        fps           : float = 0.0,
        frame_count   : int = 0,
    ) -> 'np.ndarray':
        """
        Ve toan bo overlay: bbox + TTC + panel trang thai + thanh canh bao.
        """
        import numpy as np
        import cv2

        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Ve tung track
        for t in tracks_output:
            self._draw_track(out, t)

        # 2. Panel trang thai chinh (goc tren trai)
        self._draw_main_panel(out, tracks_output, alert_level, fps)

        # 3. Thanh canh bao day du khung hinh (chi khi DANGER)
        if alert_level == AlertLevel.DANGER:
            self._draw_danger_border(out)

        # 4. Dong thoi gian + frame count (goc duoi phai)
        self._draw_footer(out, frame_count)

        return out

    def _draw_track(self, frame, t: dict):
        import cv2
        x1, y1, x2, y2 = t["bbox"]
        state  = AlertLevel(t["alert_state"])
        color  = ALERT_COLORS[state]
        thick  = 3 if state == AlertLevel.DANGER else 2

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        # Goc bounding box (visual polish)
        cl = min(18, (x2-x1)//4, (y2-y1)//4)
        corners = [
            ((x1,y1),(x1+cl,y1)), ((x1,y1),(x1,y1+cl)),
            ((x2,y1),(x2-cl,y1)), ((x2,y1),(x2,y1+cl)),
            ((x1,y2),(x1+cl,y2)), ((x1,y2),(x1,y2-cl)),
            ((x2,y2),(x2-cl,y2)), ((x2,y2),(x2,y2-cl)),
        ]
        for p1, p2 in corners:
            cv2.line(frame, p1, p2, color, 2)

        # Labels
        dist_str = f"{t['distance_m']:.1f}m"
        ttc_str  = f"TTC:{t['ttc']:.1f}s" if t.get("ttc") else f"spd:{t.get('approach_speed',0):+.1f}"
        label1   = f"{t['class_name']}  {dist_str}"
        label2   = ttc_str

        for k, lbl in enumerate([label1, label2]):
            fs = 0.52
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            lx = x1
            ly = y1 - 7 - k*(th+8)
            ly = max(ly, th+5)
            # Nen mo
            overlay = frame.copy()
            cv2.rectangle(overlay, (lx-1, ly-th-4), (lx+tw+7, ly+3), (15,15,15), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, lbl, (lx+3, ly-1),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)

        # Track ID
        cv2.putText(frame, f"#{t['track_id']}", (x2+4, y1+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    def _draw_main_panel(self, frame, tracks, level, fps):
        import cv2
        import numpy as np
        h, w = frame.shape[:2]
        pw, ph = 280, 100

        # Nen panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8+pw, 8+ph), ALERT_BG[level], -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        color = ALERT_COLORS[level]
        cv2.rectangle(frame, (8, 8), (8+pw, 8+ph), color, 2)

        # Nhan trang thai
        lbl = ALERT_LABELS_VI[level]
        cv2.putText(frame, lbl, (18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

        # Thong tin vat nguy hiem nhat
        if tracks:
            t = tracks[0]
            info1 = f"Dist: {t['distance_m']:.1f}m"
            info2 = f"TTC:  {t['ttc']:.1f}s" if t.get("ttc") else "TTC: --"
            cv2.putText(frame, info1, (18, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210,210,210), 1, cv2.LINE_AA)
            cv2.putText(frame, info2, (150, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210,210,210), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Objects: {len(tracks)}", (18, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160,160,160), 1, cv2.LINE_AA)

        # FPS
        if fps > 0:
            fps_color = (50,200,50) if fps >= 15 else (0,120,255) if fps >= 8 else (0,0,220)
            cv2.putText(frame, f"FPS: {fps:.0f}", (200, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, fps_color, 1, cv2.LINE_AA)

    def _draw_danger_border(self, frame):
        """Ve vien do nhap nhay khi DANGER."""
        import cv2
        now = time.time()
        if now - self._last_flash > self._flash_interval:
            self._flash_state = not self._flash_state
            self._last_flash  = now
        if self._flash_state:
            h, w = frame.shape[:2]
            for thickness in [4, 8]:
                cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,220), thickness)

    def _draw_footer(self, frame, frame_count: int):
        import cv2
        h, w = frame.shape[:2]
        ts = time.strftime("%H:%M:%S")
        text = f"{ts}  |  Frame #{frame_count}"
        cv2.putText(frame, text, (w-220, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,100,100), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════
#  CSV LOGGER
# ══════════════════════════════════════════════════════════
class EventLogger:
    """Ghi log su kien canh bao ra file CSV."""

    def __init__(self, log_path: str = "logs/events.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_file()

    def _init_file(self):
        if not self.log_path.exists():
            with open(self.log_path, "w") as f:
                f.write("timestamp,alert_level,distance_m,ttc_s,class_name,track_id\n")

    def log(
        self,
        alert_level : AlertLevel,
        distance_m  : float,
        ttc_s       : Optional[float],
        class_name  : str = "",
        track_id    : int = -1,
    ):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        ttc_str = f"{ttc_s:.2f}" if ttc_s is not None else ""
        row = f"{ts},{alert_level.value},{distance_m:.2f},{ttc_str},{class_name},{track_id}\n"
        with open(self.log_path, "a") as f:
            f.write(row)


# ══════════════════════════════════════════════════════════
#  DEMO & TEST
# ══════════════════════════════════════════════════════════
def demo_alert_system():
    """Demo day du AlertSystem voi cac tinh huong khac nhau."""
    print("\n" + "="*55)
    print("  DEMO HE THONG CANH BAO")
    print("="*55)

    alert = AlertSystem()
    logger = EventLogger("logs/demo_events.csv")

    scenarios = [
        (15.0, None,  "car",       "Xe o to cach 15m - binh thuong"),
        (9.0,  8.0,   "car",       "Xe o to cach 9m, TTC=8s - tiep can"),
        (6.0,  4.5,   "motorbike", "Xe may cach 6m, TTC=4.5s - CANH BAO"),
        (3.5,  2.8,   "car",       "Xe o to cach 3.5m, TTC=2.8s - CANH BAO"),
        (2.0,  1.5,   "person",    "Nguoi di bo cach 2m, TTC=1.5s - NGUY HIEM!"),
        (1.2,  0.8,   "car",       "Xe cach 1.2m, TTC=0.8s - NGUY HIEM KHAN CAP!"),
    ]

    print(f"\n  {'Tinh huong':<45} {'Muc canh bao':>14}")
    print("  " + "-"*60)

    for dist, ttc, cls, desc in scenarios:
        alert.reset_cooldowns()   # Reset cooldown de demo day du
        level = alert.trigger(dist, ttc, cls)
        icon  = {"SAFE": "  ", "WARNING": "⚡", "DANGER": "🚨"}.get(level.value, "")
        print(f"  {desc:<45} {icon} {level.value:>10}")
        logger.log(level, dist, ttc, cls)
        time.sleep(0.5)

    print(f"\n  Thong ke: {alert.get_stats()}")
    print(f"  Log luu tai: logs/demo_events.csv")
    print("\n  ✅ Demo AlertSystem hoan thanh!")


def test_tts():
    """Test phat am thanh TTS."""
    print("\n  TEST TTS AM THANH\n")
    engine = AudioEngine(lang="vi")

    test_msgs = [
        "Canh bao! Xe may cach ban 5 met!",
        "Nguy hiem! Vat can chi con 2 met, con 1 giay co the va cham!",
        "An toan. Khong co vat can phia truoc.",
    ]

    for msg in test_msgs:
        print(f"  Phat: {msg}")
        engine.speak(msg, block=True)
        time.sleep(0.5)


def calibrate_thresholds():
    """Cong cu hieu chinh nguong canh bao theo tung loai xe."""
    print("""
  HIEU CHINH NGUONG CANH BAO
  ===========================
  Nguong mac dinh:
    dist_warning = 8.0m   | ttc_warning = 5.0s
    dist_danger  = 3.0m   | ttc_danger  = 2.0s

  Khuyen nghi theo toc do:
    Thanh pho (30-50 km/h): dist_warning=6m, dist_danger=2.5m
    Quoc lo   (60-90 km/h): dist_warning=12m, dist_danger=5m
    Cao toc   (>100 km/h) : dist_warning=20m, dist_danger=8m

  De thay doi: chinh sua ALERT_CONFIG trong file nay
  hoac truyen config dict vao AlertSystem(config=...)
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="He thong canh bao")
    parser.add_argument("--demo",      action="store_true", help="Demo day du AlertSystem")
    parser.add_argument("--test-tts",  action="store_true", help="Test phat am thanh TTS")
    parser.add_argument("--calibrate", action="store_true", help="Xem huong dan hieu chinh nguong")
    args = parser.parse_args()

    if args.test_tts:
        test_tts()
    elif args.calibrate:
        calibrate_thresholds()
    else:
        demo_alert_system()
