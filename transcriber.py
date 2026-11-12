from __future__ import annotations

import itertools
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from audio_capture import TARGET_SR, AudioCaptureError, create_capture
from settings import Settings, app_data_dir

log = logging.getLogger(__name__)


def _add_nvidia_dll_dirs() -> None:
    if sys.platform != "win32":
        return
    import site

    roots = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    roots.append(str(Path(sys.prefix) / "Lib" / "site-packages"))
    seen = set()
    for root in roots:
        nvidia = Path(root) / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            p = str(bin_dir.resolve())
            if p in seen:
                continue
            seen.add(p)
            try:
                os.add_dll_directory(p)
                os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
                log.info("Добавлен путь к CUDA DLL: %s", p)
            except Exception:
                pass


_add_nvidia_dll_dirs()

_MODEL_CACHE: dict[tuple, object] = {}
_MODEL_LOCK = threading.Lock()


def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def get_model(size: str, device_pref: str = "auto"):
    from faster_whisper import WhisperModel

    attempts: list[tuple[str, str]] = []
    if device_pref in ("auto", "cuda"):
        if cuda_available():
            attempts += [("cuda", "float16"), ("cuda", "int8_float16")]
        elif device_pref == "cuda":
            log.warning("CUDA недоступна — используется CPU")
    attempts.append(("cpu", "int8"))

    last_exc: Exception | None = None
    with _MODEL_LOCK:
        for device, compute_type in attempts:
            key = (size, device, compute_type)
            if key in _MODEL_CACHE:
                return _MODEL_CACHE[key], f"{device.upper()} {compute_type}"
            try:
                log.info("Загрузка Whisper %s на %s (%s)…", size, device, compute_type)
                model = WhisperModel(
                    size,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=min(8, os.cpu_count() or 4),
                    download_root=str(app_data_dir() / "models"),
                )

                segs, _ = model.transcribe(np.zeros(TARGET_SR, dtype=np.float32),
                                           language="en", beam_size=1, vad_filter=False)
                list(segs)
                _MODEL_CACHE.clear()
                _MODEL_CACHE[key] = model
                return model, f"{device.upper()} {compute_type}"
            except Exception as exc:
                last_exc = exc
                log.warning("Не удалось загрузить модель на %s/%s: %s", device, compute_type, exc)
    raise RuntimeError(f"Не удалось загрузить модель Whisper «{size}»: {last_exc}")


class Segmenter:
    def __init__(self, max_seconds: float = 4.0, min_seconds: float = 1.0,
                 silence_seconds: float = 0.5):
        self.max_samples = int(max_seconds * TARGET_SR)
        self.min_samples = int(min_seconds * TARGET_SR)
        self.silence_samples = int(silence_seconds * TARGET_SR)
        self._blocks: list[np.ndarray] = []
        self._total = 0
        self._silent_run = 0
        self._has_speech = False
        self._noise = 0.003

    @property
    def pending(self) -> bool:
        return self._has_speech and self._total > 0

    def _threshold(self) -> float:
        return float(min(0.05, max(0.004, self._noise * 2.5)))

    def _reset(self) -> None:
        self._blocks, self._total, self._silent_run, self._has_speech = [], 0, 0, False

    def _buffer(self) -> np.ndarray:
        if not self._blocks:
            return np.zeros(0, dtype=np.float32)
        if len(self._blocks) > 1:
            self._blocks = [np.concatenate(self._blocks)]
        return self._blocks[0]

    def add(self, block: np.ndarray) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0

        if rms < self._noise * 1.5:
            self._noise = 0.95 * self._noise + 0.05 * rms
        else:
            self._noise *= 1.003
        self._noise = max(self._noise, 1e-4)

        thr = self._threshold()
        self._blocks.append(block)
        self._total += block.size
        if rms < thr:
            self._silent_run += block.size
        else:
            self._silent_run = 0
            self._has_speech = True

        if not self._has_speech:
            keep = int(0.3 * TARGET_SR)
            if self._total > keep * 2:
                buf = self._buffer()[-keep:]
                self._blocks, self._total = [buf], buf.size
            return out

        if self._total >= self.min_samples and self._silent_run >= self.silence_samples:
            buf = self._buffer()
            tail_keep = int(0.2 * TARGET_SR)
            end = max(self._total - self._silent_run + tail_keep, 0)
            out.append(buf[:end].copy())
            self._reset()
            return out

        if self._total >= self.max_samples:
            buf = self._buffer()
            cut = self._quietest_cut(buf)
            out.append(buf[:cut].copy())
            rest = buf[cut:].copy()
            self._reset()
            if rest.size:
                self._blocks, self._total, self._has_speech = [rest], rest.size, True
        return out

    def _quietest_cut(self, buf: np.ndarray) -> int:
        win = int(0.1 * TARGET_SR)
        search = int(1.5 * TARGET_SR)
        start = max(self.min_samples, buf.size - search)
        best, best_rms = buf.size, float("inf")
        for pos in range(start, buf.size - win + 1, win // 2):
            seg = buf[pos:pos + win]
            r = float(np.mean(seg * seg))
            if r < best_rms:
                best_rms, best = r, pos + win // 2
        return best

    def flush(self) -> np.ndarray | None:
        if not self.pending:
            self._reset()
            return None
        buf = self._buffer().copy()
        self._reset()
        return buf if buf.size >= int(0.4 * TARGET_SR) else None


_HALLUCINATIONS = {
    "продолжение следует", "субтитры сделал dimatorzok", "субтитры создавал dimatorzok",
    "редактор субтитров а.семкин корректор а.егорова", "спасибо за просмотр",
    "thank you for watching", "thanks for watching", "please subscribe", "you",
    "подписывайтесь на канал", "untertitel im auftrag des zdf", "sous-titrage st' 501",
    "ご視聴ありがとうございました", "字幕由amara.org社区提供", "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
}
_norm_re = re.compile(r"[\s\.\,\!\?\-—…«»\"'()\[\]]+")


def is_hallucination(text: str) -> bool:
    norm = _norm_re.sub(" ", text.lower()).strip()
    if not norm:
        return True
    if norm in _HALLUCINATIONS:
        return True
    for h in _HALLUCINATIONS:
        if len(h) > 8 and h in norm and len(norm) < len(h) + 10:
            return True

    words = norm.split()
    if len(words) >= 6 and len(set(words)) <= 2:
        return True
    return False


_ids = itertools.count(1)


class TranscriberWorker(QThread):
    utterance_ready = Signal(dict)

    status_changed = Signal(str, str)
    level_changed = Signal(float)

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.cfg = settings.copy()
        self._stop_event = threading.Event()
        self._last_level_emit = 0.0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        cfg = self.cfg
        self.status_changed.emit("loading", f"Загрузка модели Whisper «{cfg.model_size}»…")
        try:
            model, device_desc = get_model(cfg.model_size, cfg.compute_device)
        except Exception as exc:
            log.exception("Ошибка загрузки модели")
            self.status_changed.emit("error_model", f"Ошибка модели: {exc}")
            return
        if self._stop_event.is_set():
            self.status_changed.emit("stopped", "Остановлено")
            return

        capture = create_capture(cfg.audio_source, cfg.mic_device, cfg.loopback_device)
        try:
            capture.start()
        except AudioCaptureError as exc:
            log.error("Ошибка аудио: %s", exc)
            code = "error_audio"
            prefix = "Ошибка микрофона" if cfg.audio_source == "mic" else "Ошибка системного звука"
            self.status_changed.emit(code, f"{prefix}: {exc}")
            return
        except Exception as exc:
            log.exception("Неожиданная ошибка аудио")
            self.status_changed.emit("error_audio", f"Ошибка аудио: {exc}")
            return

        self.status_changed.emit("listening", f"Слушаю · {capture.description} · {device_desc}")
        segmenter = Segmenter(max_seconds=cfg.chunk_seconds)
        last_audio = time.monotonic()
        failed = False
        try:
            while not self._stop_event.is_set():
                block = capture.read(timeout=0.2)
                now = time.monotonic()
                if block is None:
                    if segmenter.pending and now - last_audio > 0.7:
                        chunk = segmenter.flush()
                        if chunk is not None:
                            self._process(model, chunk)
                    continue
                last_audio = now
                if now - self._last_level_emit > 0.1:
                    self._last_level_emit = now
                    self.level_changed.emit(float(np.sqrt(np.mean(block * block))))
                for chunk in segmenter.add(block):
                    if self._stop_event.is_set():
                        break
                    self._process(model, chunk)
        except Exception as exc:
            failed = True
            log.exception("Ошибка в потоке распознавания")
            self.status_changed.emit("error", f"Ошибка распознавания: {exc}")
        finally:
            capture.stop()
        if not failed:
            self.status_changed.emit("stopped", "Остановлено")

    def _process(self, model, audio: np.ndarray) -> None:
        if audio.size < int(0.3 * TARGET_SR):
            return
        if float(np.sqrt(np.mean(audio * audio))) < 0.002:
            return
        cfg = self.cfg
        forced = None if cfg.speech_language == "auto" else cfg.speech_language
        try:
            segments, info = model.transcribe(
                audio,
                language=forced,
                task="transcribe",
                beam_size=cfg.beam_size,
                vad_filter=cfg.vad_filter,
                vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
                condition_on_previous_text=False,
                without_timestamps=True,
                temperature=[0.0, 0.2, 0.4],
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
            texts = []
            for seg in segments:
                if seg.no_speech_prob > 0.7 and seg.avg_logprob < -0.8:
                    continue
                t = seg.text.strip()
                if t:
                    texts.append(t)
        except Exception as exc:
            log.exception("Ошибка Whisper")
            self.status_changed.emit("error", f"Ошибка распознавания: {exc}")
            return

        text = " ".join(texts).strip()
        if not text or is_hallucination(text):
            return

        lang = getattr(info, "language", None)
        prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        lang_known = bool(lang) and (forced is not None or prob >= 0.5)
        self.utterance_ready.emit({
            "id": next(_ids),
            "text": text,
            "lang": lang,
            "lang_prob": prob,
            "lang_known": lang_known,
            "ts": time.time(),
        })
