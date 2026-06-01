"""
Text-to-Speech engine for Braille recognition output.
Uses pyttsx3 (offline, free, no API key) with gTTS as cloud fallback.
"""

import io
import threading
import queue
import base64
from typing import Optional


class TTSEngine:
    """
    Thread-safe TTS engine.
    Primary: pyttsx3 (offline)
    Fallback: gTTS (online, returns audio bytes)
    """

    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()
        self._init_pyttsx3()

    def _init_pyttsx3(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 150)   # words per minute
            self._engine.setProperty("volume", 0.9)
            voices = self._engine.getProperty("voices")
            # Prefer first English voice available
            for v in voices:
                if "english" in v.name.lower() or "en" in v.id.lower():
                    self._engine.setProperty("voice", v.id)
                    break
        except Exception as e:
            print(f"[TTS] pyttsx3 init failed: {e} — will use gTTS fallback")
            self._engine = None

    def speak(self, text: str) -> bool:
        """
        Speak text aloud using pyttsx3 (blocking on the calling thread).
        Returns True on success.
        """
        if not text or not text.strip():
            return False

        if self._engine:
            try:
                with self._lock:
                    self._engine.say(text)
                    self._engine.runAndWait()
                return True
            except Exception as e:
                print(f"[TTS] pyttsx3 speak error: {e}")

        return False

    def speak_async(self, text: str):
        """Speak in a background thread so the API doesn't block."""
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()

    def synthesize_to_bytes(self, text: str, lang: str = "en") -> Optional[bytes]:
        """
        Synthesize speech to MP3 bytes (for sending over API/WebSocket).
        Uses gTTS (requires internet) for cloud-quality audio.
        Falls back to pyttsx3 wav if gTTS unavailable.
        """
        # Try gTTS first (better audio quality)
        try:
            from gtts import gTTS
            buf = io.BytesIO()
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            print(f"[TTS] gTTS failed: {e}")

        # Fallback: pyttsx3 save to file then read
        try:
            import tempfile, os
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", 150)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            eng.save_to_file(text, tmp_path)
            eng.runAndWait()
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            os.unlink(tmp_path)
            return audio_bytes
        except Exception as e:
            print(f"[TTS] pyttsx3 file save failed: {e}")
            return None

    def synthesize_to_base64(self, text: str, lang: str = "en") -> Optional[str]:
        """Return base64-encoded audio string for WebSocket/JSON transport."""
        audio_bytes = self.synthesize_to_bytes(text, lang)
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode("utf-8")
        return None

    def set_rate(self, rate: int):
        """Set speech rate (words per minute). Default 150."""
        if self._engine:
            with self._lock:
                self._engine.setProperty("rate", rate)

    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)."""
        if self._engine:
            with self._lock:
                self._engine.setProperty("volume", max(0.0, min(1.0, volume)))


# Singleton instance — shared across the app
_tts_instance: Optional[TTSEngine] = None


def get_tts_engine() -> TTSEngine:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSEngine()
    return _tts_instance
