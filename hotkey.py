from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Signal

log = logging.getLogger(__name__)

WM_HOTKEY = 0x0312
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000

_SPECIAL_VK = {
    "space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
    "pause": 0x13, "insert": 0x2D, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22, "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
}


def parse_hotkey(text: str) -> tuple[int, int]:
    mods, vk = 0, 0
    for part in (p.strip().lower() for p in text.split("+") if p.strip()):
        if part in ("ctrl", "control"):
            mods |= MOD_CONTROL
        elif part == "alt":
            mods |= MOD_ALT
        elif part == "shift":
            mods |= MOD_SHIFT
        elif part in ("win", "meta"):
            mods |= MOD_WIN
        elif len(part) == 1 and part.isalnum():
            vk = ord(part.upper())
        elif part.startswith("f") and part[1:].isdigit() and 1 <= int(part[1:]) <= 24:
            vk = 0x70 + int(part[1:]) - 1
        elif part in _SPECIAL_VK:
            vk = _SPECIAL_VK[part]
        else:
            raise ValueError(f"Неизвестная клавиша: {part}")
    if not vk:
        raise ValueError("Не указана основная клавиша")
    return mods, vk


def to_qt_sequence(text: str) -> str:
    names = {"ctrl": "Ctrl", "control": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Meta",
             "meta": "Meta"}
    return "+".join(names.get(p.lower(), p.upper() if len(p) <= 3 else p.capitalize())
                    for p in text.split("+") if p.strip())


class _Filter(QAbstractNativeEventFilter):
    def __init__(self, owner: "GlobalHotkey") -> None:
        super().__init__()
        self.owner = owner

    def nativeEventFilter(self, event_type, message):
        try:
            et = bytes(event_type.data()) if hasattr(event_type, "data") else bytes(event_type)
            if et == b"windows_generic_MSG":
                import ctypes.wintypes
                msg = ctypes.wintypes.MSG.from_address(int(message))
                if msg.message == WM_HOTKEY and msg.wParam == self.owner.hotkey_id:
                    self.owner.activated.emit()
                    return True, 0
        except Exception:
            pass
        return False, 0


class GlobalHotkey(QObject):
    activated = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app
        self.hotkey_id = 0xBEEF
        self._registered = False
        self._filter = None
        if sys.platform == "win32":
            self._filter = _Filter(self)
            app.installNativeEventFilter(self._filter)

    @property
    def registered(self) -> bool:
        return self._registered

    @property
    def supported(self) -> bool:
        return sys.platform == "win32"

    def register(self, text: str) -> bool:
        self.unregister()
        if sys.platform != "win32":
            return False
        import ctypes

        try:
            mods, vk = parse_hotkey(text)
        except ValueError as exc:
            log.warning("Горячая клавиша «%s»: %s", text, exc)
            return False
        ok = bool(ctypes.windll.user32.RegisterHotKey(None, self.hotkey_id, mods | MOD_NOREPEAT, vk))
        if not ok:
            log.warning("Сочетание «%s» уже занято другой программой", text)
        self._registered = ok
        return ok

    def unregister(self) -> None:
        if self._registered and sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.UnregisterHotKey(None, self.hotkey_id)
        self._registered = False
