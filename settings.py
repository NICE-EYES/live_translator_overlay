from __future__ import annotations

import json
import logging
import os
import sys
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger(__name__)

APP_NAME = "Live Translator Overlay"
APP_DIR_NAME = "LiveTranslatorOverlay"


def app_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    path = base / APP_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


SETTINGS_FILE = app_data_dir() / "settings.json"
CACHE_FILE = app_data_dir() / "translation_cache.json"
LOG_FILE = app_data_dir() / "app.log"

TARGET_LANGUAGES: dict[str, str] = {
    "ru": "Русский",
    "en": "Английский",
    "de": "Немецкий",
    "fr": "Французский",
    "es": "Испанский",
    "it": "Итальянский",
    "pt": "Португальский",
    "uk": "Украинский",
    "pl": "Польский",
    "cs": "Чешский",
    "nl": "Нидерландский",
    "sv": "Шведский",
    "fi": "Финский",
    "tr": "Турецкий",
    "ar": "Арабский",
    "he": "Иврит",
    "hi": "Хинди",
    "zh": "Китайский",
    "ja": "Японский",
    "ko": "Корейский",
    "id": "Индонезийский",
    "vi": "Вьетнамский",
    "th": "Тайский",
    "el": "Греческий",
    "hu": "Венгерский",
    "ro": "Румынский",
    "bg": "Болгарский",
    "da": "Датский",
}

LANGUAGE_NAMES: dict[str, str] = {
    **TARGET_LANGUAGES,
    "be": "Белорусский",
    "kk": "Казахский",
    "uz": "Узбекский",
    "ka": "Грузинский",
    "hy": "Армянский",
    "az": "Азербайджанский",
    "lt": "Литовский",
    "lv": "Латышский",
    "et": "Эстонский",
    "sr": "Сербский",
    "hr": "Хорватский",
    "sk": "Словацкий",
    "sl": "Словенский",
    "no": "Норвежский",
    "fa": "Персидский",
    "ur": "Урду",
    "bn": "Бенгальский",
    "ms": "Малайский",
    "tl": "Тагальский",
    "ca": "Каталанский",
}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo"]

AUDIO_SOURCES = {"system": "Системный звук", "mic": "Микрофон"}

TRANSLATION_MODES = {
    "local": "Локально (Argos Translate, офлайн)",
    "deepl": "DeepL API",
    "openai": "OpenAI API (или совместимый)",
}

COMPUTE_DEVICES = {
    "auto": "Авто (GPU, если доступен)",
    "cpu": "CPU",
    "cuda": "GPU (CUDA)",
}


def language_name(code: str | None) -> str:
    if not code:
        return "Язык не определён"
    return LANGUAGE_NAMES.get(code, code.upper())


@dataclass
class Settings:
    audio_source: str = "system"
    mic_device: str = ""
    loopback_device: str = ""

    model_size: str = "small"
    compute_device: str = "auto"
    speech_language: str = "auto"
    chunk_seconds: float = 4.0
    beam_size: int = 3
    vad_filter: bool = True
    live_mode: bool = True
    partial_interval: float = 0.7

    target_language: str = "ru"
    translation_mode: str = "local"
    deepl_api_key: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    api_fallback_to_local: bool = True

    opacity: int = 90
    font_size: int = 12
    max_history: int = 50
    show_original: bool = True
    compact: bool = False
    window_x: int = -1
    window_y: int = -1
    window_w: int = 460
    window_h: int = 300

    hotkey: str = "ctrl+alt+t"

    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.audio_source not in AUDIO_SOURCES:
            self.audio_source = "system"
        if self.model_size not in WHISPER_MODELS:
            self.model_size = "small"
        if self.compute_device not in COMPUTE_DEVICES:
            self.compute_device = "auto"
        if self.translation_mode not in TRANSLATION_MODES:
            self.translation_mode = "local"
        if self.target_language not in TARGET_LANGUAGES:
            self.target_language = "ru"
        self.opacity = int(min(100, max(10, int(self.opacity))))
        self.font_size = int(min(32, max(8, int(self.font_size))))
        self.max_history = int(min(1000, max(5, int(self.max_history))))
        self.chunk_seconds = float(min(5.0, max(2.0, float(self.chunk_seconds))))
        self.beam_size = int(min(5, max(1, int(self.beam_size))))
        self.partial_interval = float(min(3.0, max(0.3, float(self.partial_interval))))
        self.window_w = max(220, int(self.window_w))
        self.window_h = max(60, int(self.window_h))

    def copy(self) -> "Settings":
        return Settings(**json.loads(json.dumps(asdict(self))))


class SettingsManager:
    def __init__(self, path: Path = SETTINGS_FILE):
        self.path = path
        self._lock = threading.Lock()
        self.settings = self.load()

    def load(self) -> Settings:
        known = {f.name for f in fields(Settings)}
        data: dict = {}
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Не удалось прочитать настройки (%s), используются значения по умолчанию", exc)
            data = {}
        clean = {k: v for k, v in data.items() if k in known}
        try:
            s = Settings(**clean)
        except TypeError:
            s = Settings()
        try:
            s.validate()
        except Exception:
            s = Settings()
        return s

    def save(self) -> None:
        with self._lock:
            try:
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(asdict(self.settings), ensure_ascii=False, indent=2),
                               encoding="utf-8")
                tmp.replace(self.path)
            except Exception as exc:
                log.error("Не удалось сохранить настройки: %s", exc)
