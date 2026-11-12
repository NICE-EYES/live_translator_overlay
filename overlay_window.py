from __future__ import annotations

import html
import logging
import math
import time
from collections import deque

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QAction, QColor, QCursor, QFont, QFontMetrics, QIcon, QKeySequence,
                           QLinearGradient, QPainter, QPixmap, QShortcut)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QMenu, QMessageBox, QProgressBar, QPushButton,
                               QSizePolicy, QSlider, QSpinBox, QSystemTrayIcon, QTabWidget,
                               QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from audio_capture import list_loopback_devices, list_microphones
from hotkey import GlobalHotkey, parse_hotkey, to_qt_sequence
from settings import (APP_NAME, AUDIO_SOURCES, COMPUTE_DEVICES, LANGUAGE_NAMES, TARGET_LANGUAGES,
                      TRANSLATION_MODES, WHISPER_MODELS, Settings, SettingsManager, language_name)
from transcriber import TranscriberWorker
from translator import TranslationCache, TranslationWorker

log = logging.getLogger(__name__)

RESIZE_MARGIN = 6

STATUS_COLORS = {
    "listening": "#3ddc84",
    "loading": "#f5c542",
    "stopped": "#8a93a3",
    "error_audio": "#ff5c5c",
    "error_model": "#ff5c5c",
    "error": "#ff5c5c",
    "error_translate": "#ff9f43",
    "info": "#5cb8ff",
}

STYLE = """
QWidget { color: #e6e9ef; font-family: "Segoe UI", "Noto Sans", sans-serif; font-size: 10pt; }
#panel {
    background-color: rgba(22, 24, 30, 238);
    border: 1px solid rgba(255, 255, 255, 28);
    border-radius: 10px;
}
#titlebar, #compactbar { background: transparent; }
#appTitle { color: #c9d1dc; font-weight: 600; }
#statusText { color: #aab3c0; }
QToolButton {
    background: transparent; border: none; border-radius: 6px;
    padding: 3px 6px; color: #c9d1dc; font-size: 11pt;
}
QToolButton:hover { background: rgba(255, 255, 255, 30); }
QToolButton#closeBtn:hover { background: #c94040; color: white; }
QPushButton#startBtn {
    background: #2f7d4f; border: none; border-radius: 6px; padding: 4px 12px; font-weight: 600;
}
QPushButton#startBtn:hover { background: #379a5f; }
QPushButton#startBtn[listening="true"] { background: #a33a3a; }
QPushButton#startBtn[listening="true"]:hover { background: #c04545; }
QComboBox {
    background: rgba(255, 255, 255, 18); border: 1px solid rgba(255, 255, 255, 30);
    border-radius: 6px; padding: 2px 8px; min-height: 20px;
}
QComboBox QAbstractItemView {
    background: #22252d; selection-background-color: #3a5f9e; border: 1px solid #3a3f4b;
}
QSlider::groove:horizontal { height: 4px; background: rgba(255,255,255,40); border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; border-radius: 6px; background: #5cb8ff; }
QTextBrowser {
    background: rgba(0, 0, 0, 60); border: 1px solid rgba(255, 255, 255, 18);
    border-radius: 8px; padding: 6px;
}
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
QScrollBar::handle:vertical { background: rgba(255,255,255,60); border-radius: 3px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QProgressBar#level { background: rgba(255,255,255,25); border: none; border-radius: 2px; }
QProgressBar#level::chunk { background: #3ddc84; border-radius: 2px; }
"""

DIALOG_STYLE = """
QDialog, QTabWidget::pane, QWidget#tab { background: #1c1f26; color: #e6e9ef; }
QTabBar::tab { background: #262a33; color: #c9d1dc; padding: 6px 14px; border-top-left-radius: 6px;
               border-top-right-radius: 6px; margin-right: 2px; }
QTabBar::tab:selected { background: #34507f; color: white; }
QLabel { color: #e6e9ef; }
QLabel#hint { color: #8a93a3; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #262a33; color: #e6e9ef; border: 1px solid #3a3f4b; border-radius: 5px; padding: 3px 6px;
}
QComboBox QAbstractItemView { background: #262a33; color: #e6e9ef; selection-background-color: #3a5f9e; }
QPushButton { background: #2f3542; color: #e6e9ef; border: 1px solid #3f4656; border-radius: 5px;
              padding: 5px 12px; }
QPushButton:hover { background: #3a4252; }
QCheckBox { color: #e6e9ef; }
"""


def create_app_icon(active: bool = False) -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        grad = QLinearGradient(0, 0, size, size)
        if active:
            grad.setColorAt(0, QColor("#2fbf71"))
            grad.setColorAt(1, QColor("#1b7a48"))
        else:
            grad.setColorAt(0, QColor("#4f8df5"))
            grad.setColorAt(1, QColor("#2a4f9e"))
        p.setBrush(grad)
        p.setPen(Qt.PenStyle.NoPen)
        r = size * 0.22
        p.drawRoundedRect(0, 0, size, size, r, r)
        p.setPen(QColor("white"))
        f = QFont("Segoe UI")
        f.setBold(True)
        f.setPixelSize(int(size * 0.48))
        p.setFont(f)
        p.drawText(QRect(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, "LT")
        p.end()
        icon.addPixmap(pm)
    return icon


class StatusDot(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._color = QColor(STATUS_COLORS["stopped"])
        self.setFixedSize(12, 12)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        glow = QColor(self._color)
        glow.setAlpha(70)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(0, 0, 12, 12)
        p.setBrush(self._color)
        p.drawEllipse(2, 2, 8, 8)


class ElidedLabel(QLabel):
    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self._full = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full = text or ""
        self._update()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update()

    def _update(self) -> None:
        fm = QFontMetrics(self.font())
        super().setText(fm.elidedText(self._full, Qt.TextElideMode.ElideLeft
                                      if self.property("elideLeft") else Qt.TextElideMode.ElideRight,
                                      max(10, self.width())))


class SettingsDialog(QDialog):
    preload_requested = Signal(str)
    clear_cache_requested = Signal()

    def __init__(self, cfg: Settings, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg.copy()
        self.setWindowTitle(f"Настройки — {APP_NAME}")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setStyleSheet(DIALOG_STYLE)
        self.setMinimumWidth(520)

        tabs = QTabWidget()
        tabs.addTab(self._build_audio_tab(), "Звук и распознавание")
        tabs.addTab(self._build_translation_tab(), "Перевод")
        tabs.addTab(self._build_window_tab(), "Окно")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Сохранить")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(buttons)

    @staticmethod
    def _hint(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("hint")
        lbl.setWordWrap(True)
        return lbl

    def _build_audio_tab(self) -> QWidget:
        w = QWidget()
        w.setObjectName("tab")
        form = QFormLayout(w)

        self.source_combo = QComboBox()
        for key, name in AUDIO_SOURCES.items():
            self.source_combo.addItem(name, key)
        self.source_combo.setCurrentIndex(max(0, self.source_combo.findData(self.cfg.audio_source)))
        self.source_combo.currentIndexChanged.connect(self._fill_devices)
        form.addRow("Источник звука:", self.source_combo)

        dev_row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.device_combo.setMinimumContentsLength(28)
        refresh = QPushButton("⟳")
        refresh.setToolTip("Обновить список устройств")
        refresh.setFixedWidth(34)
        refresh.clicked.connect(self._fill_devices)
        dev_row.addWidget(self.device_combo, 1)
        dev_row.addWidget(refresh)
        form.addRow("Устройство:", dev_row)
        form.addRow("", self._hint("«Системный звук» захватывает всё, что воспроизводится через "
                                   "выбранные динамики/наушники (WASAPI loopback)."))

        self.model_combo = QComboBox()
        descr = {"tiny": "tiny — ~75 МБ, самая быстрая",
                 "base": "base — ~145 МБ, быстрая",
                 "small": "small — ~480 МБ, баланс (рекомендуется)",
                 "medium": "medium — ~1.5 ГБ, точнее, лучше с GPU",
                 "large-v3-turbo": "large-v3-turbo — ~1.6 ГБ, лучшая точность, нужна GPU"}
        for m in WHISPER_MODELS:
            self.model_combo.addItem(descr.get(m, m), m)
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(self.cfg.model_size)))
        form.addRow("Модель Whisper:", self.model_combo)

        self.device_compute = QComboBox()
        for key, name in COMPUTE_DEVICES.items():
            self.device_compute.addItem(name, key)
        self.device_compute.setCurrentIndex(max(0, self.device_compute.findData(self.cfg.compute_device)))
        form.addRow("Вычисления:", self.device_compute)

        self.speech_lang = QComboBox()
        self.speech_lang.addItem("Автоопределение (для каждой реплики)", "auto")
        for code, name in sorted(LANGUAGE_NAMES.items(), key=lambda kv: kv[1]):
            self.speech_lang.addItem(name, code)
        self.speech_lang.setCurrentIndex(max(0, self.speech_lang.findData(self.cfg.speech_language)))
        form.addRow("Язык речи:", self.speech_lang)

        self.chunk_spin = QDoubleSpinBox()
        self.chunk_spin.setRange(2.0, 5.0)
        self.chunk_spin.setSingleStep(0.5)
        self.chunk_spin.setSuffix(" с")
        self.chunk_spin.setValue(self.cfg.chunk_seconds)
        form.addRow("Макс. длина фрагмента:", self.chunk_spin)

        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 5)
        self.beam_spin.setValue(self.cfg.beam_size)
        form.addRow("Beam size:", self.beam_spin)

        self.vad_check = QCheckBox("Отсекать тишину и шум (Silero VAD)")
        self.vad_check.setChecked(self.cfg.vad_filter)
        self.live_check = QCheckBox("Мгновенный режим: показывать текст и перевод во время речи")
        self.live_check.setChecked(self.cfg.live_mode)
        form.addRow("", self.vad_check)
        form.addRow("", self.live_check)
        form.addRow("", self._hint("Меньше фрагмент и beam size → меньше задержка; больше → точнее. "
                                   "Аудио всегда обрабатывается только на этом компьютере."))
        self._fill_devices()
        return w

    def _build_translation_tab(self) -> QWidget:
        w = QWidget()
        w.setObjectName("tab")
        form = QFormLayout(w)

        self.target_combo = QComboBox()
        for code, name in TARGET_LANGUAGES.items():
            self.target_combo.addItem(name, code)
        self.target_combo.setCurrentIndex(max(0, self.target_combo.findData(self.cfg.target_language)))
        form.addRow("Переводить на:", self.target_combo)

        self.mode_combo = QComboBox()
        for key, name in TRANSLATION_MODES.items():
            self.mode_combo.addItem(name, key)
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(self.cfg.translation_mode)))
        self.mode_combo.currentIndexChanged.connect(self._update_mode_fields)
        form.addRow("Режим перевода:", self.mode_combo)

        self.deepl_key = QLineEdit(self.cfg.deepl_api_key)
        self.deepl_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.deepl_key.setPlaceholderText("xxxxxxxx-xxxx-…:fx")
        form.addRow("DeepL API-ключ:", self.deepl_key)

        self.openai_key = QLineEdit(self.cfg.openai_api_key)
        self.openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_key.setPlaceholderText("sk-…")
        form.addRow("OpenAI API-ключ:", self.openai_key)

        self.openai_model = QLineEdit(self.cfg.openai_model)
        form.addRow("Модель OpenAI:", self.openai_model)
        self.openai_url = QLineEdit(self.cfg.openai_base_url)
        form.addRow("Base URL:", self.openai_url)

        self.show_keys = QCheckBox("Показать ключи")
        self.show_keys.toggled.connect(self._toggle_keys)
        form.addRow("", self.show_keys)

        self.fallback_check = QCheckBox("При ошибке API переводить офлайн (Argos)")
        self.fallback_check.setChecked(self.cfg.api_fallback_to_local)
        form.addRow("", self.fallback_check)

        form.addRow("", self._hint(
            "В режиме API в интернет отправляется только распознанный ТЕКСТ, звук — никогда. "
            "Для локального режима языковые пакеты скачиваются один раз (по ~50–100 МБ), "
            "затем перевод работает без интернета."))

        btns = QHBoxLayout()
        preload = QPushButton("Скачать офлайн-пакеты")
        preload.setToolTip("Скачать пакеты Argos для перевода с основных языков на выбранный")
        preload.clicked.connect(self._on_preload)
        clear = QPushButton("Очистить кэш переводов")
        clear.clicked.connect(self.clear_cache_requested.emit)
        btns.addWidget(preload)
        btns.addWidget(clear)
        btns.addStretch(1)
        form.addRow("", btns)
        self._update_mode_fields()
        return w

    def _build_window_tab(self) -> QWidget:
        w = QWidget()
        w.setObjectName("tab")
        form = QFormLayout(w)

        op_row = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(self.cfg.opacity)
        self.opacity_label = QLabel(f"{self.cfg.opacity}%")
        self.opacity_label.setFixedWidth(40)
        self.opacity_slider.valueChanged.connect(lambda v: self.opacity_label.setText(f"{v}%"))
        op_row.addWidget(self.opacity_slider, 1)
        op_row.addWidget(self.opacity_label)
        form.addRow("Прозрачность окна:", op_row)

        self.font_spin = QSpinBox()
        self.font_spin.setRange(8, 32)
        self.font_spin.setSuffix(" pt")
        self.font_spin.setValue(self.cfg.font_size)
        form.addRow("Размер шрифта:", self.font_spin)

        self.history_spin = QSpinBox()
        self.history_spin.setRange(5, 1000)
        self.history_spin.setValue(self.cfg.max_history)
        form.addRow("Строк в истории:", self.history_spin)

        self.original_check = QCheckBox("Показывать оригинальный текст")
        self.original_check.setChecked(self.cfg.show_original)
        form.addRow("", self.original_check)

        self.hotkey_edit = QLineEdit(self.cfg.hotkey)
        self.hotkey_edit.setPlaceholderText("ctrl+alt+t")
        form.addRow("Горячая клавиша Старт/Стоп:", self.hotkey_edit)
        form.addRow("", self._hint("Формат: ctrl+alt+t, ctrl+shift+f9, alt+space … "
                                   "Работает глобально, даже если окно не в фокусе."))
        return w

    def _fill_devices(self) -> None:
        source = self.source_combo.currentData()
        current = self.cfg.mic_device if source == "mic" else self.cfg.loopback_device
        self.device_combo.clear()
        default = "Микрофон по умолчанию" if source == "mic" else "Динамики по умолчанию (Windows)"
        self.device_combo.addItem(default, "")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            names = list_microphones() if source == "mic" else list_loopback_devices()
        finally:
            QApplication.restoreOverrideCursor()
        for n in names:
            self.device_combo.addItem(n, n)
        if current and self.device_combo.findData(current) < 0:
            self.device_combo.addItem(f"{current} (не найдено)", current)
        self.device_combo.setCurrentIndex(max(0, self.device_combo.findData(current)))

    def _update_mode_fields(self) -> None:
        mode = self.mode_combo.currentData()
        self.deepl_key.setEnabled(mode == "deepl")
        for wdg in (self.openai_key, self.openai_model, self.openai_url):
            wdg.setEnabled(mode == "openai")
        self.fallback_check.setEnabled(mode != "local")

    def _toggle_keys(self, show: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if show else QLineEdit.EchoMode.Password
        self.deepl_key.setEchoMode(mode)
        self.openai_key.setEchoMode(mode)

    def _on_preload(self) -> None:
        tgt = self.target_combo.currentData()
        ans = QMessageBox.question(
            self, "Офлайн-пакеты",
            f"Скачать пакеты для перевода с английского, немецкого, французского, испанского, "
            f"итальянского, китайского и японского на «{TARGET_LANGUAGES[tgt]}»?\n\n"
            "Потребуется интернет и около 0.5–1 ГБ места. Загрузка идёт в фоне.")
        if ans == QMessageBox.StandardButton.Yes:
            self.preload_requested.emit(tgt)

    def _on_accept(self) -> None:
        hk = self.hotkey_edit.text().strip().lower() or "ctrl+alt+t"
        try:
            parse_hotkey(hk)
        except ValueError as exc:
            QMessageBox.warning(self, "Горячая клавиша", f"Неверное сочетание: {exc}")
            return
        mode = self.mode_combo.currentData()
        if mode == "deepl" and not self.deepl_key.text().strip():
            QMessageBox.warning(self, "DeepL", "Укажите API-ключ DeepL или выберите локальный режим.")
            return
        if (mode == "openai" and "api.openai.com" in self.openai_url.text()
                and not self.openai_key.text().strip()):
            QMessageBox.warning(self, "OpenAI", "Укажите API-ключ OpenAI или выберите локальный режим.")
            return

        c = self.cfg
        c.audio_source = self.source_combo.currentData()
        dev = self.device_combo.currentData() or ""
        if c.audio_source == "mic":
            c.mic_device = dev
        else:
            c.loopback_device = dev
        c.model_size = self.model_combo.currentData()
        c.compute_device = self.device_compute.currentData()
        c.speech_language = self.speech_lang.currentData()
        c.chunk_seconds = float(self.chunk_spin.value())
        c.beam_size = int(self.beam_spin.value())
        c.vad_filter = self.vad_check.isChecked()
        c.live_mode = self.live_check.isChecked()
        c.target_language = self.target_combo.currentData()
        c.translation_mode = mode
        c.deepl_api_key = self.deepl_key.text().strip()
        c.openai_api_key = self.openai_key.text().strip()
        c.openai_model = self.openai_model.text().strip() or "gpt-4.1-mini"
        c.openai_base_url = self.openai_url.text().strip() or "https://api.openai.com/v1"
        c.api_fallback_to_local = self.fallback_check.isChecked()
        c.opacity = int(self.opacity_slider.value())
        c.font_size = int(self.font_spin.value())
        c.max_history = int(self.history_spin.value())
        c.show_original = self.original_check.isChecked()
        c.hotkey = hk
        c.validate()
        self.accept()

    def result_settings(self) -> Settings:
        return self.cfg


RESTART_KEYS = ("live_mode", "audio_source", "mic_device", "loopback_device", "model_size", "compute_device",
                "speech_language", "chunk_seconds", "beam_size", "vad_filter")


class OverlayWindow(QWidget):
    def __init__(self, settings_manager: SettingsManager) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.sm = settings_manager
        self.cfg = settings_manager.settings
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(create_app_icon())
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setMinimumSize(240, 44)
        self.setStyleSheet(STYLE)

        self.entries: deque[dict] = deque(maxlen=self.cfg.max_history)
        self.transcriber: TranscriberWorker | None = None
        self._restart_pending = False
        self._quitting = False
        self._rec_state = ("stopped", "Остановлено")
        self._tr_error = ""
        self._info = ""
        self._tray_hint_shown = False

        self.cache = TranslationCache()
        self.translation_worker = TranslationWorker(self.cache, self.cfg)
        self.translation_worker.translated.connect(self._on_translated)
        self.translation_worker.failed.connect(self._on_translate_failed)
        self.translation_worker.status_changed.connect(self._on_translation_status)
        self.translation_worker.start()

        self._render_timer = QTimer(self, singleShot=True, interval=40)
        self._render_timer.timeout.connect(self._render_history)
        self._save_timer = QTimer(self, singleShot=True, interval=600)
        self._save_timer.timeout.connect(self._save_geometry)
        self._info_timer = QTimer(self, singleShot=True, interval=5000)
        self._info_timer.timeout.connect(self._clear_info)

        self._build_ui()
        self._build_tray()
        self._setup_hotkey()
        self._restore_geometry()
        self.setWindowOpacity(self.cfg.opacity / 100.0)
        self._apply_font()
        self._set_compact(self.cfg.compact, save=False)
        self._refresh_status()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(RESIZE_MARGIN, RESIZE_MARGIN, RESIZE_MARGIN, RESIZE_MARGIN)
        self.panel = QFrame()
        self.panel.setObjectName("panel")
        self.panel.setCursor(Qt.CursorShape.ArrowCursor)
        outer.addWidget(self.panel)
        lay = QVBoxLayout(self.panel)
        lay.setContentsMargins(8, 4, 8, 8)
        lay.setSpacing(6)

        self.titlebar = QWidget()
        self.titlebar.setObjectName("titlebar")
        tb = QHBoxLayout(self.titlebar)
        tb.setContentsMargins(0, 0, 0, 0)
        tb.setSpacing(6)
        self.status_dot = StatusDot()
        self.status_text = ElidedLabel("Остановлено")
        self.status_text.setObjectName("statusText")
        self.level_bar = QProgressBar()
        self.level_bar.setObjectName("level")
        self.level_bar.setRange(0, 100)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedSize(46, 5)
        self.level_bar.setToolTip("Уровень входного сигнала")
        tb.addWidget(self.status_dot)
        tb.addWidget(self.status_text, 1)
        tb.addWidget(self.level_bar)
        self.compact_btn = self._tool("▭", "Компактный режим", lambda: self._set_compact(True))
        self.hide_btn = self._tool("—", "Скрыть окно в трей", self.hide_to_tray)
        self.close_btn = self._tool("✕", "Выход", self.quit_app)
        self.close_btn.setObjectName("closeBtn")
        for b in (self.compact_btn, self.hide_btn, self.close_btn):
            tb.addWidget(b)
        lay.addWidget(self.titlebar)

        self.toolbar = QWidget()
        tl = QHBoxLayout(self.toolbar)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(6)
        self.start_btn = QPushButton("▶ Старт")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setToolTip(f"Начать/остановить прослушивание ({self.cfg.hotkey})")
        self.start_btn.clicked.connect(self.toggle_listening)
        tl.addWidget(self.start_btn)

        self.lang_combo = QComboBox()
        self.lang_combo.setToolTip("Язык перевода")
        for code, name in TARGET_LANGUAGES.items():
            self.lang_combo.addItem(f"→ {name}", code)
        self.lang_combo.setCurrentIndex(max(0, self.lang_combo.findData(self.cfg.target_language)))
        self.lang_combo.currentIndexChanged.connect(self._on_target_changed)
        tl.addWidget(self.lang_combo, 1)

        op_icon = QLabel("◐")
        op_icon.setToolTip("Прозрачность")
        tl.addWidget(op_icon)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(self.cfg.opacity)
        self.opacity_slider.setFixedWidth(70)
        self.opacity_slider.setToolTip(f"Прозрачность: {self.cfg.opacity}%")
        self.opacity_slider.valueChanged.connect(self._on_opacity)
        tl.addWidget(self.opacity_slider)

        tl.addWidget(self._tool("🗑", "Очистить текст", self.clear_history))
        tl.addWidget(self._tool("⚙", "Настройки", self.open_settings))
        lay.addWidget(self.toolbar)

        self.history = QTextBrowser()
        self.history.setOpenLinks(False)
        self.history.setReadOnly(True)
        self.history.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.history.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.history.setPlaceholderText("Нажмите «▶ Старт» или Ctrl+Alt+T — здесь появится перевод.")
        lay.addWidget(self.history, 1)

        self.compactbar = QWidget()
        self.compactbar.setObjectName("compactbar")
        cb = QHBoxLayout(self.compactbar)
        cb.setContentsMargins(0, 0, 0, 0)
        cb.setSpacing(6)
        self.compact_dot = StatusDot()
        self.compact_text = ElidedLabel("…")
        self.compact_text.setProperty("elideLeft", True)
        self.compact_text.setToolTip("Двойной щелчок — развернуть")
        self.compact_start = self._tool("▶", "Старт/Стоп", self.toggle_listening)
        cb.addWidget(self.compact_dot)
        cb.addWidget(self.compact_text, 1)
        cb.addWidget(self.compact_start)
        cb.addWidget(self._tool("▢", "Развернуть", lambda: self._set_compact(False)))
        cb.addWidget(self._tool("—", "Скрыть окно в трей", self.hide_to_tray))
        lay.addWidget(self.compactbar)

        for w in (self.titlebar, self.status_text, self.compactbar, self.compact_text, self.toolbar):
            w.installEventFilter(self)

        self._local_shortcut = QShortcut(QKeySequence(to_qt_sequence(self.cfg.hotkey)), self)
        self._local_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._local_shortcut.activated.connect(self.toggle_listening)

    def _tool(self, text: str, tip: str, slot) -> QToolButton:
        b = QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(slot)
        return b

    def _build_tray(self) -> None:
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(create_app_icon(), self)
        self.tray.setToolTip(APP_NAME)
        menu = QMenu()
        self.act_show = QAction("Показать окно", self, triggered=self.show_from_tray)
        self.act_toggle = QAction("▶ Старт", self, triggered=self.toggle_listening)
        act_compact = QAction("Компактный режим", self,
                              triggered=lambda: (self.show_from_tray(), self._set_compact(not self.cfg.compact)))
        act_settings = QAction("Настройки…", self, triggered=self.open_settings)
        act_quit = QAction("Выход", self, triggered=self.quit_app)
        for a in (self.act_show, self.act_toggle, act_compact, act_settings):
            menu.addAction(a)
        menu.addSeparator()
        menu.addAction(act_quit)
        self._tray_menu = menu
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _setup_hotkey(self) -> None:
        self.hotkey = GlobalHotkey(QApplication.instance(), self)
        self.hotkey.activated.connect(self.toggle_listening)
        self._register_hotkey()

    def _register_hotkey(self) -> None:
        if self.hotkey.supported and not self.hotkey.register(self.cfg.hotkey):
            self._show_info(f"Сочетание {self.cfg.hotkey} занято другой программой — "
                            "выберите другое в настройках")
        self._local_shortcut.setKey(QKeySequence(to_qt_sequence(self.cfg.hotkey)))

        self._local_shortcut.setEnabled(not self.hotkey.registered)
        self.start_btn.setToolTip(f"Начать/остановить прослушивание ({self.cfg.hotkey})")

    def _restore_geometry(self) -> None:
        c = self.cfg
        self.resize(c.window_w, c.window_h)
        screen = QApplication.primaryScreen().availableGeometry()
        pos = QPoint(c.window_x, c.window_y)
        on_screen = any(s.availableGeometry().contains(pos + QPoint(20, 20))
                        for s in QApplication.screens())
        if c.window_x < 0 or c.window_y < 0 or not on_screen:
            pos = QPoint(screen.right() - c.window_w - 30, screen.bottom() - c.window_h - 30)
        self.move(pos)

    def _save_geometry(self) -> None:
        g = self.geometry()
        self.cfg.window_x, self.cfg.window_y = g.x(), g.y()
        self.cfg.window_w = g.width()
        if not self.cfg.compact:
            self.cfg.window_h = g.height()
        self.sm.save()

    def moveEvent(self, e):
        super().moveEvent(e)
        self._save_timer.start()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._save_timer.start()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 2))

    def _edges_at(self, pos: QPoint):
        m = RESIZE_MARGIN + 2
        r = self.rect()
        found = []
        if pos.x() <= m:
            found.append(Qt.Edge.LeftEdge)
        elif pos.x() >= r.width() - m:
            found.append(Qt.Edge.RightEdge)
        if not self.cfg.compact:
            if pos.y() <= m:
                found.append(Qt.Edge.TopEdge)
            elif pos.y() >= r.height() - m:
                found.append(Qt.Edge.BottomEdge)
        if not found:
            return None
        edges = found[0]
        for e in found[1:]:
            edges = edges | e
        return edges

    @staticmethod
    def _cursor_for(edges) -> Qt.CursorShape:
        L, R, T, B = Qt.Edge.LeftEdge, Qt.Edge.RightEdge, Qt.Edge.TopEdge, Qt.Edge.BottomEdge
        if edges in (L | T, R | B):
            return Qt.CursorShape.SizeFDiagCursor
        if edges in (R | T, L | B):
            return Qt.CursorShape.SizeBDiagCursor
        if edges in (L, R):
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.SizeVerCursor

    def mouseMoveEvent(self, e):
        edges = self._edges_at(e.position().toPoint())
        if edges is None:
            self.unsetCursor()
        else:
            self.setCursor(self._cursor_for(edges))
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            edges = self._edges_at(e.position().toPoint())
            handle = self.windowHandle()
            if edges is not None and handle is not None:
                handle.startSystemResize(edges)
                return
            if handle is not None:
                handle.startSystemMove()
                return
        super().mousePressEvent(e)

    def eventFilter(self, obj, event):
        et = event.type()
        if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            if obj in (self.titlebar, self.status_text, self.compactbar, self.compact_text, self.toolbar):
                handle = self.windowHandle()
                if handle is not None:
                    handle.startSystemMove()
                    return True
        elif et == QEvent.Type.MouseButtonDblClick and obj in (self.compactbar, self.compact_text):
            self._set_compact(False)
            return True
        elif et == QEvent.Type.MouseButtonDblClick and obj in (self.titlebar, self.status_text):
            self._set_compact(True)
            return True
        return super().eventFilter(obj, event)

    def _set_compact(self, compact: bool, save: bool = True) -> None:
        if save and not self.cfg.compact and compact:
            self.cfg.window_h = self.height()
        self.cfg.compact = compact
        self.titlebar.setVisible(not compact)
        self.toolbar.setVisible(not compact)
        self.history.setVisible(not compact)
        self.compactbar.setVisible(compact)
        if compact:
            fm = QFontMetrics(self.compact_text.font())
            h = max(28, fm.height() + 10) + 2 * RESIZE_MARGIN + 12
            self.setMinimumHeight(h)
            self.setMaximumHeight(h)
            self.resize(self.width(), h)
            self._update_compact_text()
        else:
            self.setMinimumHeight(140)
            self.setMaximumHeight(16777215)
            self.resize(self.width(), max(self.cfg.window_h, 180))
        if save:
            self._save_geometry()

    def _update_compact_text(self) -> None:
        if not self.entries:
            self.compact_text.setText("Перевод появится здесь")
            return
        e = self.entries[-1]
        if e.get("translation"):
            self.compact_text.setText(e["translation"])
        else:
            self.compact_text.setText(e["text"] + "  …")
        self.compact_text.setToolTip(f"{e['text']}\n\n{e.get('translation') or ''}\n\n"
                                     "Двойной щелчок — развернуть")

    def _on_tray_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                      QSystemTrayIcon.ActivationReason.DoubleClick):
            if self.isVisible():
                self.raise_()
            else:
                self.show_from_tray()

    def hide_to_tray(self) -> None:
        if self.tray is None:
            self.showMinimized()
            return
        self.hide()
        if not self._tray_hint_shown:
            self._tray_hint_shown = True
            self.tray.showMessage(APP_NAME, "Окно скрыто. Программа продолжает работать — "
                                            "щёлкните по иконке в трее, чтобы вернуть окно.",
                                  QSystemTrayIcon.MessageIcon.Information, 3000)

    def show_from_tray(self) -> None:
        self.show()
        self.raise_()

    @property
    def listening(self) -> bool:
        return self.transcriber is not None

    def toggle_listening(self) -> None:
        if self.listening:
            self.stop_listening()
        else:
            self.start_listening()

    def start_listening(self) -> None:
        if self.transcriber is not None:
            return
        self._tr_error = ""
        w = TranscriberWorker(self.cfg)
        w.utterance_ready.connect(self._on_utterance)
        w.status_changed.connect(self._on_rec_status)
        w.level_changed.connect(self._on_level)
        w.finished.connect(self._on_worker_finished)
        self.transcriber = w
        w.start()
        self._update_start_buttons()

    def stop_listening(self) -> None:
        if self.transcriber is None:
            return
        self.transcriber.stop()
        self.start_btn.setEnabled(False)
        self.start_btn.setText("… Остановка")

    def _on_worker_finished(self) -> None:
        w = self.sender()
        if w is not None and w is not self.transcriber:
            return
        if self.transcriber is not None:
            self.transcriber.deleteLater()
        self.transcriber = None
        self.level_bar.setValue(0)
        self._update_start_buttons()
        if self._restart_pending and not self._quitting:
            self._restart_pending = False
            self.start_listening()

    def _update_start_buttons(self) -> None:
        on = self.listening
        self.start_btn.setEnabled(True)
        self.start_btn.setText("■ Стоп" if on else "▶ Старт")
        self.start_btn.setProperty("listening", on)
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self.compact_start.setText("■" if on else "▶")
        if self.tray is not None:
            self.act_toggle.setText("■ Стоп" if on else "▶ Старт")
            self.tray.setIcon(create_app_icon(active=on))

    def _on_rec_status(self, code: str, text: str) -> None:
        self._rec_state = (code, text)
        self._refresh_status()

    def _on_level(self, rms: float) -> None:
        db = 20 * math.log10(max(rms, 1e-6))
        self.level_bar.setValue(int(max(0.0, min(1.0, (db + 60) / 60)) * 100))

    def _on_translation_status(self, code: str, text: str) -> None:
        if code == "error_translate":
            self._tr_error = text
        elif code == "ok":
            if self._tr_error:
                self._tr_error = ""
        elif code == "info":
            self._show_info(text)
            return
        self._refresh_status()

    def _show_info(self, text: str) -> None:
        self._info = text
        self._info_timer.start()
        self._refresh_status()

    def _clear_info(self) -> None:
        self._info = ""
        self._refresh_status()

    def _refresh_status(self) -> None:
        code, text = self._rec_state
        color = STATUS_COLORS.get(code, "#8a93a3")
        shown = text
        if self._tr_error and code in ("listening", "stopped"):
            color, shown = STATUS_COLORS["error_translate"], self._tr_error
        elif self._info and code in ("listening", "stopped"):
            color, shown = STATUS_COLORS["info"], self._info
        self.status_dot.set_color(color)
        self.compact_dot.set_color(color)
        self.status_text.setText(shown)
        self.status_text.setToolTip(shown)
        self.compact_dot.setToolTip(shown)
        if self.tray is not None:
            self.tray.setToolTip(f"{APP_NAME}\n{shown[:100]}")

    def _on_utterance(self, u: dict) -> None:
        if self._rec_state[0] == "error" and self.listening:
            self._rec_state = ("listening", "Слушаю")
            self._refresh_status()
        e = self._find_entry(u["id"])
        if u.get("discard"):
            if e is not None:
                self.entries.remove(e)
                self._schedule_render()
            return
        if e is None:
            e = dict(u, translation=None, engine="", error="")
            self.entries.append(e)
        else:
            e.update(text=u["text"], lang=u["lang"], lang_prob=u["lang_prob"],
                     lang_known=u["lang_known"], final=u.get("final", True), error="")
        self.translation_worker.submit(u["id"], u["text"], u.get("lang"),
                                       self.cfg.target_language, u.get("final", True))
        self._schedule_render()

    def _find_entry(self, uid: int) -> dict | None:
        for e in reversed(self.entries):
            if e["id"] == uid:
                return e
        return None

    def _on_translated(self, uid: int, text: str, engine: str) -> None:
        e = self._find_entry(uid)
        if e is not None:
            e["translation"], e["engine"], e["error"] = text, engine, ""
            self._schedule_render()

    def _on_translate_failed(self, uid: int, msg: str) -> None:
        e = self._find_entry(uid)
        if e is not None:
            e["error"] = msg
            self._schedule_render()

    def _schedule_render(self) -> None:
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _entry_html(self, e: dict) -> str:
        fs = self.cfg.font_size
        esc = html.escape
        t = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
        if e["lang_known"]:
            lang = f"{language_name(e['lang'])}"
            lang_color = "#5cb8ff"
        else:
            lang = "Язык не определён"
            if e.get("lang"):
                lang += f" (возможно, {language_name(e['lang'])} {e['lang_prob'] * 100:.0f}%)"
            lang_color = "#ff9f43"
        live = not e.get("final", True)
        engine = f" · {esc(e['engine'])}" if e.get("engine") and not live else ""
        if live:
            engine = ' · <span style="color:#3ddc84;">● говорит…</span>'
        parts = [
            f'<p style="margin:0; font-size:{max(7, fs - 3)}pt;">'
            f'<span style="color:#6f7a8c;">{t}</span>&nbsp;&nbsp;'
            f'<span style="color:{lang_color};">● {esc(lang)}</span>'
            f'<span style="color:#55606f;">{engine}</span></p>'
        ]
        if self.cfg.show_original:
            parts.append(f'<p style="margin:1px 0 0 0; color:#a9b3c1; font-size:{max(7, fs - 1)}pt;">'
                         f'{esc(e["text"])}</p>')
        if e.get("translation"):
            color = "#cfd6e0" if live else "#ffffff"
            tr = f'<span style="color:{color};">{esc(e["translation"])}</span>'
        elif e.get("error"):
            tr = (f'<span style="color:#ff9f43;">⚠ перевод не удался: {esc(e["error"])}</span>'
                  + ("" if self.cfg.show_original else
                     f'<br><span style="color:#a9b3c1;">{esc(e["text"])}</span>'))
        else:
            tr = '<span style="color:#6f7a8c;">перевод…</span>'
        parts.append(f'<p style="margin:2px 0 10px 0; font-size:{fs + 1}pt; font-weight:600;">{tr}</p>')
        return "".join(parts)

    def _render_history(self) -> None:
        sb = self.history.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        old_value = sb.value()
        body = "".join(self._entry_html(e) for e in self.entries)
        self.history.setHtml(f"<html><body>{body}</body></html>")
        if at_bottom:
            sb.setValue(sb.maximum())
        else:
            sb.setValue(old_value)
        self._update_compact_text()

    def clear_history(self) -> None:
        self.entries.clear()
        self.history.clear()
        self._update_compact_text()

    def _apply_font(self) -> None:
        f = self.compact_text.font()
        f.setPointSize(self.cfg.font_size)
        f.setBold(True)
        self.compact_text.setFont(f)
        self._schedule_render()

    def _on_target_changed(self) -> None:
        code = self.lang_combo.currentData()
        if code and code != self.cfg.target_language:
            self.cfg.target_language = code
            self.translation_worker.update_settings(self.cfg)
            self.sm.save()

    def _on_opacity(self, v: int) -> None:
        self.cfg.opacity = v
        self.setWindowOpacity(v / 100.0)
        self.opacity_slider.setToolTip(f"Прозрачность: {v}%")
        self._save_timer.start()

    def open_settings(self) -> None:
        self.show_from_tray()
        dlg = SettingsDialog(self.cfg, self)
        dlg.preload_requested.connect(self._preload_packages)
        dlg.clear_cache_requested.connect(self._clear_cache)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.apply_settings(dlg.result_settings())

    def _preload_packages(self, tgt: str) -> None:
        for src in ("en", "de", "fr", "es", "it", "zh", "ja"):
            if src != tgt:
                self.translation_worker.preload(src, tgt)
        self._show_info("Загрузка офлайн-пакетов запущена в фоне…")

    def _clear_cache(self) -> None:
        self.cache.clear()
        self._show_info("Кэш переводов очищен")

    def apply_settings(self, new: Settings) -> None:
        old = self.cfg
        restart = self.listening and any(getattr(old, k) != getattr(new, k) for k in RESTART_KEYS)
        hotkey_changed = old.hotkey != new.hotkey

        geometry = (old.window_x, old.window_y, old.window_w, old.window_h, old.compact)
        for k, v in vars(new).items():
            setattr(old, k, v)
        old.window_x, old.window_y, old.window_w, old.window_h, old.compact = geometry
        self.sm.save()

        self.translation_worker.update_settings(self.cfg)
        self.setWindowOpacity(self.cfg.opacity / 100.0)
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(self.cfg.opacity)
        self.opacity_slider.blockSignals(False)
        self.lang_combo.blockSignals(True)
        self.lang_combo.setCurrentIndex(max(0, self.lang_combo.findData(self.cfg.target_language)))
        self.lang_combo.blockSignals(False)
        if self.entries.maxlen != self.cfg.max_history:
            self.entries = deque(self.entries, maxlen=self.cfg.max_history)
        self._apply_font()
        if self.cfg.compact:
            self._set_compact(True, save=False)
        if hotkey_changed:
            self._register_hotkey()
        if restart:
            self._restart_pending = True
            self.stop_listening()

    def quit_app(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self._save_geometry()
        try:
            self.hotkey.unregister()
        except Exception:
            pass
        if self.transcriber is not None:
            self.transcriber.stop()
            self.transcriber.wait(5000)
        self.translation_worker.stop()
        self.translation_worker.wait(3000)
        self.cache.save()
        self.sm.save()
        if self.tray is not None:
            self.tray.hide()
        QApplication.quit()

    def closeEvent(self, e):
        if not self._quitting and self.tray is not None:
            e.ignore()
            self.hide_to_tray()
        else:
            super().closeEvent(e)

    def sizeHint(self) -> QSize:
        return QSize(self.cfg.window_w, self.cfg.window_h)
