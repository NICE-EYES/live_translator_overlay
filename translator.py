from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Signal

from settings import CACHE_FILE, Settings, language_name

log = logging.getLogger(__name__)


class TranslationError(Exception):
    pass


ENGLISH_NAMES = {
    "ru": "Russian", "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "uk": "Ukrainian", "pl": "Polish", "cs": "Czech",
    "nl": "Dutch", "sv": "Swedish", "fi": "Finnish", "tr": "Turkish", "ar": "Arabic",
    "he": "Hebrew", "hi": "Hindi", "zh": "Chinese (Simplified)", "ja": "Japanese",
    "ko": "Korean", "id": "Indonesian", "vi": "Vietnamese", "th": "Thai", "el": "Greek",
    "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian", "da": "Danish",
}

ARGOS_CODE_MAP = {"no": "nb", "nn": "nb", "jw": "jv", "iw": "he"}


class TranslationCache:
    def __init__(self, path: Path = CACHE_FILE, max_items: int = 5000) -> None:
        self.path = path
        self.max_items = max_items
        self._data: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._dirty = 0
        self.load()

    @staticmethod
    def key(mode: str, src: str | None, tgt: str, text: str) -> str:
        norm = re.sub(r"\s+", " ", text).strip()
        return f"{mode}|{src or 'auto'}|{tgt}|{norm}"

    def get(self, key: str) -> str | None:
        with self._lock:
            val = self._data.get(key)
            if val is not None:
                self._data.move_to_end(key)
            return val

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)
            self._dirty += 1
        if self._dirty >= 50:
            self.save()

    def load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._data = OrderedDict(list(data.items())[-self.max_items:])
        except Exception as exc:
            log.warning("Кэш переводов не загружен: %s", exc)

    def save(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            snapshot = dict(self._data)
            self._dirty = 0
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)
        except Exception as exc:
            log.warning("Кэш переводов не сохранён: %s", exc)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._dirty = 1
        self.save()


class ArgosBackend:
    label = "Argos (офлайн)"

    def __init__(self, status_cb: Callable[[str], None] | None = None) -> None:
        self._status = status_cb or (lambda msg: None)
        self._index_updated = False
        self._unavailable: dict[tuple[str, str], float] = {}

    @staticmethod
    def code(c: str | None) -> str:
        c = (c or "en").lower()
        return ARGOS_CODE_MAP.get(c, c)

    @staticmethod
    def _has_pair(src: str, tgt: str) -> bool:
        import argostranslate.translate as at
        a = at.get_language_from_code(src)
        b = at.get_language_from_code(tgt)
        return bool(a and b and a.get_translation(b) is not None)

    def ensure_pair(self, src: str, tgt: str) -> None:
        import argostranslate.package as ap
        import argostranslate.settings as asettings
        import argostranslate.translate as at

        if self._has_pair(src, tgt):
            return
        key = (src, tgt)
        if time.time() - self._unavailable.get(key, 0) < 300:
            raise TranslationError(f"Нет офлайн-модели {src}→{tgt}")

        self._status(f"Загрузка языкового пакета {language_name(src)} → {language_name(tgt)}…")
        try:
            if not self._index_updated:
                ap.update_package_index()
                self._index_updated = True
            if not Path(asettings.local_package_index).exists():
                raise TranslationError("Нет интернета: не удалось получить список языковых пакетов")
            available = ap.get_available_packages()
            installed = {(p.from_code, p.to_code) for p in ap.get_installed_packages()}

            def find(a: str, b: str):
                return next((p for p in available if p.from_code == a and p.to_code == b), None)

            needed = []
            direct = find(src, tgt)
            if direct is not None:
                needed = [direct]
            else:
                for a, b in ((src, "en"), ("en", tgt)):
                    if a == b or (a, b) in installed:
                        continue
                    pkg = find(a, b)
                    if pkg is None:
                        raise TranslationError(
                            f"Argos не поддерживает пару {language_name(src)} → {language_name(tgt)}")
                    needed.append(pkg)
            for pkg in needed:
                log.info("Установка пакета Argos %s→%s", pkg.from_code, pkg.to_code)
                ap.install_from_path(pkg.download())
            at.get_installed_languages.cache_clear()
        except TranslationError:
            self._unavailable[key] = time.time()
            raise
        except Exception as exc:
            self._unavailable[key] = time.time()
            raise TranslationError(f"Не удалось скачать языковой пакет: {exc}") from exc

        if not self._has_pair(src, tgt):
            self._unavailable[key] = time.time()
            raise TranslationError(f"Пара {src}→{tgt} недоступна офлайн")

    def translate(self, text: str, src: str | None, tgt: str) -> str:
        try:
            import argostranslate.translate as at
        except Exception as exc:
            raise TranslationError(f"argostranslate не установлен ({exc}). "
                                   "Выполните: pip install argostranslate") from exc

        s, t = self.code(src), self.code(tgt)
        if s == t:
            return text
        self.ensure_pair(s, t)
        try:
            return at.translate(text, s, t).strip()
        except Exception as exc:
            raise TranslationError(f"Ошибка Argos: {exc}") from exc


DEEPL_SOURCE = {"ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "he", "hu",
                "id", "it", "ja", "ko", "lt", "lv", "nb", "nl", "pl", "pt", "ro", "ru", "sk",
                "sl", "sv", "th", "tr", "uk", "vi", "zh"}
DEEPL_TARGET_MAP = {"en": "EN-US", "pt": "PT-BR", "zh": "ZH-HANS"}


class DeepLBackend:
    label = "DeepL"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key.strip()

    def translate(self, text: str, src: str | None, tgt: str) -> str:
        import requests

        if not self.api_key:
            raise TranslationError("Не указан API-ключ DeepL")
        url = ("https://api-free.deepl.com/v2/translate" if self.api_key.endswith(":fx")
               else "https://api.deepl.com/v2/translate")
        payload = {"text": [text], "target_lang": DEEPL_TARGET_MAP.get(tgt, tgt.upper())}
        if src and src in DEEPL_SOURCE:
            payload["source_lang"] = src.upper()
        try:
            r = requests.post(url, json=payload, timeout=10,
                              headers={"Authorization": f"DeepL-Auth-Key {self.api_key}"})
        except Exception as exc:
            raise TranslationError(f"DeepL недоступен: {exc}") from exc
        if r.status_code == 403:
            raise TranslationError("DeepL: неверный API-ключ")
        if r.status_code == 456:
            raise TranslationError("DeepL: исчерпан лимит символов")
        if r.status_code == 429:
            raise TranslationError("DeepL: слишком много запросов")
        if r.status_code >= 400:
            raise TranslationError(f"DeepL: HTTP {r.status_code} {r.text[:200]}")
        try:
            return r.json()["translations"][0]["text"].strip()
        except Exception as exc:
            raise TranslationError(f"DeepL: неожиданный ответ ({exc})") from exc


class OpenAIBackend:
    label = "OpenAI"

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip() or "gpt-4.1-mini"
        self.base_url = (base_url.strip() or "https://api.openai.com/v1").rstrip("/")

    def translate(self, text: str, src: str | None, tgt: str) -> str:
        import requests

        is_official = "api.openai.com" in self.base_url
        if is_official and not self.api_key:
            raise TranslationError("Не указан API-ключ OpenAI")
        target = ENGLISH_NAMES.get(tgt, tgt)
        source = ENGLISH_NAMES.get(src or "", "auto-detect")
        system = (
            "You are a real-time subtitle translator. "
            f"Translate the user's message into {target}. Source language: {source}. "
            "The text is a fragment of live speech recognition and may be incomplete or contain "
            "recognition errors — translate the intended meaning naturally. "
            "Reply with the translation only: no quotes, notes or explanations."
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": text}],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                              headers=headers, timeout=20)
        except Exception as exc:
            raise TranslationError(f"OpenAI недоступен: {exc}") from exc
        if r.status_code == 401:
            raise TranslationError("OpenAI: неверный API-ключ")
        if r.status_code == 429:
            raise TranslationError("OpenAI: превышен лимит запросов или нет средств")
        if r.status_code >= 400:
            raise TranslationError(f"OpenAI: HTTP {r.status_code} {r.text[:200]}")
        try:
            return r.json()["choices"][0]["message"]["content"].strip().strip('"«»')
        except Exception as exc:
            raise TranslationError(f"OpenAI: неожиданный ответ ({exc})") from exc


class Translator:
    def __init__(self, cache: TranslationCache, status_cb: Callable[[str], None] | None = None):
        self.cache = cache
        self.argos = ArgosBackend(status_cb)
        self.cfg = Settings()
        self._api = None

    def configure(self, cfg: Settings) -> None:
        self.cfg = cfg.copy()
        if cfg.translation_mode == "deepl":
            self._api = DeepLBackend(cfg.deepl_api_key)
        elif cfg.translation_mode == "openai":
            self._api = OpenAIBackend(cfg.openai_api_key, cfg.openai_model, cfg.openai_base_url)
        else:
            self._api = None

    def translate(self, text: str, src: str | None, tgt: str) -> tuple[str, str]:
        if src and ArgosBackend.code(src) == ArgosBackend.code(tgt):
            return text, "без перевода"
        mode = self.cfg.translation_mode
        key = TranslationCache.key(mode, src, tgt, text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached, "кэш"

        if self._api is not None:
            try:
                result = self._api.translate(text, src, tgt)
                self.cache.put(key, result)
                return result, self._api.label
            except TranslationError as api_exc:
                if not self.cfg.api_fallback_to_local:
                    raise
                log.warning("API-перевод не удался (%s), пробую офлайн", api_exc)
                try:
                    result = self.argos.translate(text, src, tgt)
                except TranslationError as local_exc:
                    raise TranslationError(f"{api_exc}; офлайн: {local_exc}") from local_exc

                self.cache.put(TranslationCache.key("local", src, tgt, text), result)
                return result, f"{self.argos.label}, т.к. {api_exc}"

        result = self.argos.translate(text, src, tgt)
        self.cache.put(key, result)
        return result, self.argos.label


class TranslationWorker(QThread):
    translated = Signal(int, str, str)
    failed = Signal(int, str)
    status_changed = Signal(str, str)

    MAX_BACKLOG = 20

    def __init__(self, cache: TranslationCache, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._pending_cfg: Settings | None = settings.copy()
        self._cfg_lock = threading.Lock()
        self.translator = Translator(cache, lambda m: self.status_changed.emit("info", m))

    def update_settings(self, settings: Settings) -> None:
        with self._cfg_lock:
            self._pending_cfg = settings.copy()

    def submit(self, uid: int, text: str, src: str | None, tgt: str) -> None:
        self._queue.put(("translate", uid, text, src, tgt))

    def preload(self, src: str, tgt: str) -> None:
        self._queue.put(("preload", 0, "", src, tgt))

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                break
            with self._cfg_lock:
                if self._pending_cfg is not None:
                    self.translator.configure(self._pending_cfg)
                    self._pending_cfg = None

            while self._queue.qsize() > self.MAX_BACKLOG:
                try:
                    old = self._queue.get_nowait()
                except queue.Empty:
                    break
                if old is None:
                    return
                if old[0] == "translate":
                    self.failed.emit(old[1], "пропущено: перевод не успевает")

            kind, uid, text, src, tgt = job
            try:
                if kind == "preload":
                    self.translator.argos.ensure_pair(ArgosBackend.code(src), ArgosBackend.code(tgt))
                    self.status_changed.emit("info", f"Офлайн-пакет {language_name(src)} → "
                                                     f"{language_name(tgt)} установлен")
                    continue
                result, engine = self.translator.translate(text, src, tgt)
                self.translated.emit(uid, result, engine)
                self.status_changed.emit("ok", "")
            except TranslationError as exc:
                self.failed.emit(uid, str(exc))
                self.status_changed.emit("error_translate", f"Ошибка перевода: {exc}")
            except Exception as exc:
                log.exception("Непредвиденная ошибка перевода")
                self.failed.emit(uid, str(exc))
                self.status_changed.emit("error_translate", f"Ошибка перевода: {exc}")
        self.translator.cache.save()
