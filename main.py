from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler

from settings import APP_NAME, LOG_FILE, SettingsManager


def setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    try:
        fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass
    if sys.stderr is not None:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("stanza").setLevel(logging.WARNING)


def install_excepthook() -> None:
    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logging.getLogger("unhandled").error(
            "Необработанное исключение:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))

    sys.excepthook = hook


def set_windows_app_id() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("LiveTranslatorOverlay.App")
        except Exception:
            pass


def main() -> int:
    setup_logging()
    install_excepthook()
    set_windows_app_id()
    log = logging.getLogger("main")
    log.info("Запуск %s", APP_NAME)

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setQuitOnLastWindowClosed(False)

    from overlay_window import OverlayWindow, create_app_icon

    app.setWindowIcon(create_app_icon())
    settings_manager = SettingsManager()
    window = OverlayWindow(settings_manager)
    window.show()
    app.aboutToQuit.connect(settings_manager.save)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
