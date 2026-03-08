"""
audio_bridge.py — The "Harmonic Bridge"
Bridges the gap between raw audio and the HFF.
Two responsibilities:
  1. Spectral resonance — FFT-based frequency analysis that lets Alivai
     "hear" vocal timbre (low-band warmth vs high-band crispness).
  2. Speech-to-text — Whisper transcription that converts voice into text
     compatible with the existing chat pipeline.
"""

import io
import struct
import math
import threading
import time
import wave
import numpy as np
import requests as http_client

from voice_interface import VoiceInterface, SAMPLE_RATE, BLOCK_SIZE
from audio_processor import HarmonicAudioProcessor


# ── Defaults ─────────────────────────────────────────────────────────────────
OLLAMA_BASE = "http://localhost:11434"
WHISPER_ENDPOINT = "http://localhost:8000/v1/chat/completions"  # self — transcribed text routes here
LOW_BAND = (20, 250)       # fundamental tones, "thrum"
HIGH_BAND = (2000, 8000)   # sibilance, crispness


class ResonanceAudioBridge:
    """
    Orchestrates mic → VAD → FFT resonance + Whisper STT → chat pipeline.
    The spectral resonance tuple (delta, zeta) captures the *feel* of the voice.
    The transcribed text captures the *meaning*.
    """

    def __init__(
        self,
        hff_engine=None,
        chat_callback=None,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        smoothing: float = 0.85,
        vad_threshold: float = 0.01,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.smoothing = smoothing
        self.hff_engine = hff_engine
        self._chat_callback = chat_callback

        # Sub-components
        self._mic = VoiceInterface(sample_rate=sample_rate, block_size=block_size)
        self._processor = HarmonicAudioProcessor(
            sample_rate=sample_rate,
            block_size=block_size,
            vad_threshold=vad_threshold,
        )

        # Spectral resonance state
        self._resonance_lock = threading.Lock()
        self._resonance = (0.0, 0.0)  # (delta_input, zeta_input)
        self._analysis_buffer: list[bytes] = []
        self._analysis_thread = None
        self._running = False

        # Wire pipeline: mic → processor → bridge
        self._mic.register_callback(self._on_mic_chunk)
        self._processor.register_phrase_callback(self._on_phrase_complete)

    # ── Public API ───────────────────────────────────────────────────────

    def start(self):
        """Start the full audio pipeline: mic → VAD → resonance + STT."""
        if self._running:
            return
        self._running = True
        self._mic.start()
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()

    def stop(self):
        """Stop the audio pipeline and release resources."""
        self._running = False
        self._mic.stop()
        if self._analysis_thread:
            self._analysis_thread.join(timeout=2)
            self._analysis_thread = None
        self._processor.reset()
        self._analysis_buffer.clear()

    def terminate(self):
        """Fully release all resources."""
        self.stop()
        self._mic.terminate()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def resonance(self) -> tuple[float, float]:
        """Current spectral resonance: (delta_input, zeta_input)."""
        with self._resonance_lock:
            return self._resonance

    @property
    def is_speaking(self) -> bool:
        return self._processor.is_speaking

    # ── Pipeline Callbacks ───────────────────────────────────────────────

    def _on_mic_chunk(self, chunk: bytes):
        """Called by VoiceInterface for every raw audio frame."""
        # Feed to VAD processor
        self._processor.feed(chunk)
        # Accumulate for spectral analysis
        self._analysis_buffer.append(chunk)

    def _on_phrase_complete(self, phrase_bytes: bytes):
        """Called by AudioProcessor when a complete spoken phrase is detected."""
        # Transcribe in a separate thread to avoid blocking the mic
        threading.Thread(
            target=self._transcribe_and_dispatch,
            args=(phrase_bytes,),
            daemon=True,
        ).start()

    # ── Spectral Resonance (FFT Analysis) ────────────────────────────────

    def _analysis_loop(self):
        """
        Periodically analyze accumulated audio for spectral resonance.
        Runs every ~200 ms — extracts low/high frequency band power.
        """
        while self._running:
            time.sleep(0.2)
            if not self._analysis_buffer:
                continue

            # Consume buffer
            chunks = list(self._analysis_buffer)
            self._analysis_buffer.clear()

            raw = b"".join(chunks)
            n_samples = len(raw) // 2
            if n_samples < 64:
                continue

            # Decode 16-bit PCM → float array
            samples = struct.unpack(f"<{n_samples}h", raw)
            signal = np.array(samples, dtype=np.float64) / 32768.0

            # FFT
            spectrum = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.sample_rate)

            # Band power
            low_mask = (freqs >= LOW_BAND[0]) & (freqs <= LOW_BAND[1])
            high_mask = (freqs >= HIGH_BAND[0]) & (freqs <= HIGH_BAND[1])

            low_power = np.mean(spectrum[low_mask]) if np.any(low_mask) else 0.0
            high_power = np.mean(spectrum[high_mask]) if np.any(high_mask) else 0.0

            total = low_power + high_power + 1e-30
            zeta_input = low_power / total     # fundamental dominance
            delta_input = high_power / total   # sibilance dominance

            # Exponential smoothing
            with self._resonance_lock:
                old_d, old_z = self._resonance
                new_d = self.smoothing * old_d + (1.0 - self.smoothing) * delta_input
                new_z = self.smoothing * old_z + (1.0 - self.smoothing) * zeta_input
                self._resonance = (new_d, new_z)

            # Push spectral signature to HFF if engine is attached
            if self.hff_engine:
                try:
                    self.hff_engine.process_resonance(zeta_input)
                except Exception:
                    pass

    # ── Speech-to-Text (Whisper via Ollama) ──────────────────────────────

    def _transcribe_and_dispatch(self, phrase_bytes: bytes):
        """
        Convert phrase audio to WAV, transcribe with Whisper, and dispatch
        the resulting text through the chat pipeline.
        """
        wav_buffer = self._pcm_to_wav(phrase_bytes)
        transcript = self._whisper_transcribe(wav_buffer)
        if not transcript or not transcript.strip():
            return

        # Get current resonance signature at time of utterance
        spectral = self.resonance

        # Dispatch through the chat callback (or POST to own endpoint)
        if self._chat_callback:
            self._chat_callback(transcript, spectral)

    def _whisper_transcribe(self, wav_buffer: io.BytesIO) -> str:
        """
        Transcribe WAV audio using a local Whisper model.
        Tries Ollama's Whisper endpoint first, falls back to
        faster-whisper / whisper CLI if available.
        """
        try:
            # Try local faster-whisper first (most common local install)
            return self._transcribe_faster_whisper(wav_buffer)
        except ImportError:
            pass

        try:
            # Fallback: OpenAI-compatible Whisper API (e.g. local whisper server)
            return self._transcribe_api(wav_buffer)
        except Exception:
            pass

        return ""

    def _transcribe_faster_whisper(self, wav_buffer: io.BytesIO) -> str:
        """Transcribe using faster-whisper (local, no API call)."""
        from faster_whisper import WhisperModel
        if not hasattr(self, "_whisper_model"):
            self._whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        wav_buffer.seek(0)
        segments, _ = self._whisper_model.transcribe(wav_buffer, language="en")
        return " ".join(seg.text.strip() for seg in segments)

    def _transcribe_api(self, wav_buffer: io.BytesIO) -> str:
        """Transcribe via OpenAI-compatible /v1/audio/transcriptions endpoint."""
        wav_buffer.seek(0)
        resp = http_client.post(
            "http://localhost:8000/v1/audio/transcriptions",
            files={"file": ("phrase.wav", wav_buffer, "audio/wav")},
            data={"model": "whisper-1"},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json().get("text", "")
        return ""

    # ── Utility ──────────────────────────────────────────────────────────

    def _pcm_to_wav(self, pcm_bytes: bytes) -> io.BytesIO:
        """Wrap raw 16-bit PCM bytes into a proper WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_bytes)
        buf.seek(0)
        return buf
