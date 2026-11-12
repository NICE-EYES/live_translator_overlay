from __future__ import annotations

import logging
import queue
import sys
import threading

import numpy as np

log = logging.getLogger(__name__)

TARGET_SR = 16000
BLOCK_SECONDS = 0.1


class AudioCaptureError(Exception):
    pass


class StreamResampler:
    def __init__(self, sr_in: int, sr_out: int = TARGET_SR):
        self.sr_in = int(sr_in)
        self.sr_out = int(sr_out)
        self.ratio = self.sr_in / self.sr_out
        r = round(self.ratio)
        self.int_ratio = r if (r >= 1 and abs(self.ratio - r) < 1e-9) else 0
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.sr_in == self.sr_out:
            return x.astype(np.float32, copy=False)
        buf = np.concatenate([self._buf, x.astype(np.float32, copy=False)])
        if self.int_ratio:
            r = self.int_ratio
            n = (len(buf) // r) * r
            out = buf[:n].reshape(-1, r).mean(axis=1)
            self._buf = buf[n:]
            return out.astype(np.float32)
        last = len(buf) - 1
        if last < 1:
            self._buf = buf
            return np.zeros(0, dtype=np.float32)
        positions = np.arange(self._pos, last, self.ratio)
        if len(positions) == 0:
            self._buf = buf
            return np.zeros(0, dtype=np.float32)
        out = np.interp(positions, np.arange(len(buf)), buf).astype(np.float32)
        next_pos = positions[-1] + self.ratio
        drop = min(int(next_pos), last)
        self._buf = buf[drop:]
        self._pos = next_pos - drop
        return out


def to_mono(data: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1:
        return data.reshape(-1)
    return data.reshape(-1, channels).mean(axis=1)


class BaseCapture:
    description = "Источник звука"

    def __init__(self) -> None:
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=600)
        self._resampler: StreamResampler | None = None
        self._lock = threading.Lock()
        self.running = False

    def _push(self, mono: np.ndarray) -> None:
        if self._resampler is None:
            return
        try:
            out = self._resampler.process(mono)
        except Exception:
            return
        if out.size == 0:
            return
        try:
            self._queue.put_nowait(out)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(out)
            except queue.Empty:
                pass

    def read(self, timeout: float = 0.2) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


def _preferred_hostapi_index(sd) -> int | None:
    try:
        for i, api in enumerate(sd.query_hostapis()):
            if "WASAPI" in api["name"]:
                return i
    except Exception:
        pass
    return None


def list_microphones() -> list[str]:
    try:
        import sounddevice as sd
    except Exception as exc:
        log.warning("sounddevice недоступен: %s", exc)
        return []
    try:
        devices = sd.query_devices()
    except Exception as exc:
        log.warning("Не удалось получить список микрофонов: %s", exc)
        return []
    api = _preferred_hostapi_index(sd)
    names: list[str] = []
    for d in devices:
        if d.get("max_input_channels", 0) <= 0:
            continue
        if api is not None and d.get("hostapi") != api:
            continue
        name = d["name"]

        if name not in names:
            names.append(name)
    return names


class MicrophoneCapture(BaseCapture):
    description = "Микрофон"

    def __init__(self, device_name: str = "") -> None:
        super().__init__()
        self.device_name = device_name
        self._stream = None

    def _resolve_device(self, sd):
        if not self.device_name:
            return None
        api = _preferred_hostapi_index(sd)
        candidates = []
        for idx, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0 and d["name"] == self.device_name:
                candidates.append((0 if d.get("hostapi") == api else 1, idx))
        if not candidates:
            log.warning("Микрофон «%s» не найден, используется устройство по умолчанию",
                        self.device_name)
            return None
        return sorted(candidates)[0][1]

    def start(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            raise AudioCaptureError(f"Библиотека sounddevice не загружена: {exc}") from exc

        try:
            device = self._resolve_device(sd)
            info = sd.query_devices(device, "input")
        except Exception as exc:
            raise AudioCaptureError(f"Микрофон не найден: {exc}") from exc

        samplerate = int(info.get("default_samplerate") or 48000)
        max_ch = max(1, int(info.get("max_input_channels") or 1))

        extra = None
        try:
            hostapi_name = sd.query_hostapis(info["hostapi"])["name"]
            if "WASAPI" in hostapi_name:
                extra = sd.WasapiSettings(exclusive=False)
        except Exception:
            extra = None

        last_error: Exception | None = None
        for channels in sorted({1, min(2, max_ch)}):
            try:
                self._channels = channels
                self._resampler = StreamResampler(samplerate, TARGET_SR)
                self._stream = sd.InputStream(
                    device=device,
                    channels=channels,
                    samplerate=samplerate,
                    dtype="float32",
                    blocksize=int(samplerate * BLOCK_SECONDS),
                    latency="low",
                    extra_settings=extra,
                    callback=self._callback,
                )
                self._stream.start()
                self.running = True
                self.description = f"Микрофон: {info['name']}"
                log.info("Микрофон открыт: %s, %d Гц, %d кан.", info["name"], samplerate, channels)
                return
            except Exception as exc:
                last_error = exc
                self._stream = None
        raise AudioCaptureError(f"Не удалось открыть микрофон: {last_error}")

    def _callback(self, indata, frames, time_info, status) -> None:
        self._push(to_mono(indata, self._channels).copy())

    def stop(self) -> None:
        self.running = False
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                log.debug("Ошибка при закрытии микрофона: %s", exc)
        self.clear()


def list_loopback_devices() -> list[str]:
    if sys.platform != "win32":
        return []
    try:
        import pyaudiowpatch as pyaudio
    except Exception as exc:
        log.warning("PyAudioWPatch недоступен: %s", exc)
        return []
    pa = pyaudio.PyAudio()
    try:
        return [d["name"] for d in pa.get_loopback_device_info_generator()]
    except Exception as exc:
        log.warning("Не удалось получить список loopback-устройств: %s", exc)
        return []
    finally:
        pa.terminate()


class LoopbackCapture(BaseCapture):
    description = "Системный звук"

    def __init__(self, device_name: str = "") -> None:
        super().__init__()
        self.device_name = device_name
        self._pa = None
        self._stream = None
        self._channels = 2

    def _find_device(self, pyaudio) -> dict:
        pa = self._pa
        if self.device_name:
            for dev in pa.get_loopback_device_info_generator():
                if dev["name"] == self.device_name:
                    return dev
            log.warning("Loopback-устройство «%s» не найдено, используются динамики по умолчанию",
                        self.device_name)

        try:
            return pa.get_default_wasapi_loopback()
        except Exception:
            pass
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if speakers.get("isLoopbackDevice"):
            return speakers
        for dev in pa.get_loopback_device_info_generator():
            if speakers["name"] in dev["name"]:
                return dev
        raise AudioCaptureError("Не найдено loopback-устройство для динамиков по умолчанию")

    def start(self) -> None:
        if sys.platform != "win32":
            raise AudioCaptureError("Захват системного звука (WASAPI loopback) доступен только в Windows")
        try:
            import pyaudiowpatch as pyaudio
        except Exception as exc:
            raise AudioCaptureError(
                f"PyAudioWPatch не установлен ({exc}). Выполните: pip install PyAudioWPatch") from exc

        try:
            self._pa = pyaudio.PyAudio()
            dev = self._find_device(pyaudio)
            self._channels = max(1, int(dev["maxInputChannels"]))
            samplerate = int(dev["defaultSampleRate"])
            self._resampler = StreamResampler(samplerate, TARGET_SR)
            self._paContinue = pyaudio.paContinue
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=self._channels,
                rate=samplerate,
                input=True,
                input_device_index=dev["index"],
                frames_per_buffer=int(samplerate * BLOCK_SECONDS),
                stream_callback=self._callback,
            )
            self._stream.start_stream()
            self.running = True
            self.description = f"Системный звук: {dev['name']}"
            log.info("Loopback открыт: %s, %d Гц, %d кан.", dev["name"], samplerate, self._channels)
        except AudioCaptureError:
            self.stop()
            raise
        except Exception as exc:
            self.stop()
            raise AudioCaptureError(f"Не удалось захватить системный звук: {exc}") from exc

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            data = np.frombuffer(in_data, dtype=np.float32)
            self._push(to_mono(data, self._channels))
        except Exception:
            pass
        return (None, self._paContinue)

    def stop(self) -> None:
        self.running = False
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as exc:
                log.debug("Ошибка при закрытии loopback: %s", exc)
        pa, self._pa = self._pa, None
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
        self.clear()


def create_capture(source: str, mic_device: str = "", loopback_device: str = "") -> BaseCapture:
    if source == "mic":
        return MicrophoneCapture(mic_device)
    return LoopbackCapture(loopback_device)
