"""
audio_processor.py — The "Neural Pre-Processor"
Strips noise to protect the HFF from Logic Drift.
Implements VAD (Voice Activity Detection), buffer management, and
normalization to produce clean phrase-level audio for transcription.
"""

import struct
import math
import webrtcvad

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_BLOCK_SIZE = 480       # 30 ms @ 16 kHz
DEFAULT_VAD_MODE = 2           # 0-3, higher = more aggressive filtering
DEFAULT_VAD_THRESHOLD = 0.01   # RMS floor — below this counts as silence
DEFAULT_MAX_SILENCE_SEC = 0.5  # seconds of silence before phrase ends


class HarmonicAudioProcessor:
    """
    Accumulates raw byte chunks from VoiceInterface, applies RMS-based
    VAD gating, and emits complete phrases as contiguous byte buffers.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = DEFAULT_BLOCK_SIZE,
        vad_threshold: float = DEFAULT_VAD_THRESHOLD,
        max_silence_seconds: float = DEFAULT_MAX_SILENCE_SEC,
        vad_mode: int = DEFAULT_VAD_MODE,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.vad_threshold = vad_threshold
        self.max_silence_frames = int(max_silence_seconds / (block_size / sample_rate))

        # webrtcvad for secondary confirmation
        self._vad = webrtcvad.Vad(vad_mode)

        # State
        self._phrase_buffer: list[bytes] = []
        self._silence_counter = 0
        self._is_speaking = False
        self._on_phrase_complete = None

    # ── Public API ───────────────────────────────────────────────────────

    def register_phrase_callback(self, callback):
        """Register callable(bytes) that receives a completed phrase buffer."""
        self._on_phrase_complete = callback

    def feed(self, chunk: bytes):
        """
        Accept a raw audio chunk from VoiceInterface.
        Applies VAD gating and accumulates until phrase boundary detected.
        """
        rms = self._rms(chunk)
        is_voice = rms >= self.vad_threshold

        # Secondary check with webrtcvad (expects 16-bit PCM, 16 kHz, 30 ms frames)
        try:
            webrtc_voice = self._vad.is_speech(chunk, self.sample_rate)
        except Exception:
            webrtc_voice = False

        voice_detected = is_voice or webrtc_voice

        if voice_detected:
            if not self._is_speaking:
                self._is_speaking = True
                self._silence_counter = 0
            self._phrase_buffer.append(chunk)
            self._silence_counter = 0
        elif self._is_speaking:
            # Still accumulate during short silences (breathing, pauses)
            self._phrase_buffer.append(chunk)
            self._silence_counter += 1
            if self._silence_counter >= self.max_silence_frames:
                self._emit_phrase()

    def reset(self):
        """Clear the buffer and reset state."""
        self._phrase_buffer.clear()
        self._silence_counter = 0
        self._is_speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    # ── Internal ─────────────────────────────────────────────────────────

    def _emit_phrase(self):
        """Concatenate buffered chunks and fire the phrase-complete callback."""
        if self._phrase_buffer and self._on_phrase_complete:
            phrase_bytes = b"".join(self._phrase_buffer)
            self._on_phrase_complete(phrase_bytes)
        self._phrase_buffer.clear()
        self._silence_counter = 0
        self._is_speaking = False

    @staticmethod
    def _rms(chunk: bytes) -> float:
        """Calculate Root Mean Square of 16-bit PCM audio."""
        n_samples = len(chunk) // 2
        if n_samples == 0:
            return 0.0
        samples = struct.unpack(f"<{n_samples}h", chunk)
        sum_sq = sum(s * s for s in samples)
        return math.sqrt(sum_sq / n_samples) / 32768.0  # normalize to [0, 1]
