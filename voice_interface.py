"""
voice_interface.py — The "Ear"
Maintains an open PyAudio stream that captures raw byte chunks from the
system's default input device. Polls for incoming sound pressure levels
and feeds chunks to the AudioProcessor pipeline.
"""

import threading
import pyaudio

# ── Audio Constants ──────────────────────────────────────────────────────────
SAMPLE_RATE = 16000          # 16 kHz — standard for STT models
CHANNELS = 1                 # mono
FORMAT = pyaudio.paInt16     # 16-bit PCM
BLOCK_SIZE = 480             # 30 ms frames at 16 kHz (required by webrtcvad)


class VoiceInterface:
    """
    Stream-based microphone interface.  Opens the default system input
    device and continuously feeds raw byte chunks to a registered callback.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = BLOCK_SIZE):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._running = False
        self._callback = None
        self._thread = None

    # ── Public API ───────────────────────────────────────────────────────

    def register_callback(self, callback):
        """Register a callable(bytes) that receives each raw audio chunk."""
        self._callback = callback

    def start(self):
        """Open the mic stream and begin capturing on a background thread."""
        if self._running:
            return
        self._running = True
        self._stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.block_size,
        )
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop capturing and release the stream."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def terminate(self):
        """Release PyAudio resources entirely."""
        self.stop()
        if self._pa:
            self._pa.terminate()
            self._pa = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Internal ─────────────────────────────────────────────────────────

    def _capture_loop(self):
        """Continuously read from the mic and push chunks to the callback."""
        while self._running and self._stream:
            try:
                data = self._stream.read(self.block_size, exception_on_overflow=False)
                if self._callback:
                    self._callback(data)
            except IOError:
                # Buffer overflow under load — skip frame, keep running
                continue
            except Exception:
                break
