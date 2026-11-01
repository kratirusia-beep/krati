"""
Audio Signal Processing & Acoustic Environment Classifier.
Epic 6: Audio Signal Processing & Classroom Acoustic Analysis (Task 11 & 12)
Computes real-time RMS, VAD (Voice Activity Detection), and acoustic dynamics.
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np

from src.config import get_config

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False


@dataclass
class AudioAcousticState:
    rms_energy: float
    zero_crossing_rate: float
    is_speech_active: bool
    acoustic_environment: str  # 'INSTRUCTOR_MONOLOGUE', 'CLASSROOM_DISCUSSION', 'SIDE_CHATTER', 'SILENT_WORK'
    speech_duration_sec: float
    chatter_index: float       # 0.0 to 1.0
    audio_engagement_factor: float # 0.0 to 1.0


class AudioAnalyzer:
    """Non-blocking audio capture and classroom acoustic feature extraction."""

    def __init__(self, use_microphone: bool = True):
        self.config = get_config()
        self.sample_rate = self.config.get("audio.sample_rate", 16000)
        self.silence_threshold = self.config.get("audio.silence_rms_threshold", 0.015)
        self.vad_threshold = self.config.get("audio.voice_activity_threshold", 0.040)
        self.chatter_threshold = self.config.get("audio.classroom_chatter_threshold", 0.090)

        self.use_microphone = use_microphone and HAS_SOUNDDEVICE
        self.stream: Optional[Any] = None
        self.lock = threading.Lock()
        self.is_running = False

        # Current state
        self.current_rms = 0.02
        self.current_zcr = 0.05
        self.is_voice_active = False
        self.consecutive_speech_sec = 0.0
        self.last_audio_tick = time.time()
        self.current_state = AudioAcousticState(
            rms_energy=0.02,
            zero_crossing_rate=0.05,
            is_speech_active=False,
            acoustic_environment="SILENT_WORK",
            speech_duration_sec=0.0,
            chatter_index=0.0,
            audio_engagement_factor=0.75,
        )

    def start(self):
        """Starts non-blocking background microphone capture thread or simulation."""
        self.is_running = True
        if self.use_microphone:
            try:
                blocksize = int(self.sample_rate * 0.5)  # 500ms chunks
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    blocksize=blocksize,
                    callback=self._audio_callback,
                )
                self.stream.start()
                print("[AudioAnalyzer] Real-time microphone capture started.")
                return
            except Exception as e:
                print(f"[AudioAnalyzer] Could not open microphone ({e}). Switching to simulation mode.")
                self.use_microphone = False

        # Simulated dynamic background audio thread if no microphone or disabled
        self.sim_thread = threading.Thread(target=self._simulated_audio_loop, daemon=True)
        self.sim_thread.start()
        print("[AudioAnalyzer] Simulated audio environment loop active.")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any):
        if status:
            pass
        audio_chunk = indata[:, 0]
        self._process_chunk(audio_chunk)

    def _process_chunk(self, audio_data: np.ndarray):
        now = time.time()
        dt = max(0.001, now - self.last_audio_tick)
        self.last_audio_tick = now

        if len(audio_data) == 0:
            return

        # 1. Root-Mean-Square (RMS) Energy
        rms = float(np.sqrt(np.mean(audio_data**2) + 1e-12))

        # 2. Zero Crossing Rate (ZCR)
        zero_crossings = np.sum(np.abs(np.diff(audio_data > 0)))
        zcr = float(zero_crossings / max(1, len(audio_data)))

        # 3. Voice Activity Detection (VAD)
        is_speech = rms >= self.vad_threshold

        if is_speech:
            self.consecutive_speech_sec += dt
        else:
            self.consecutive_speech_sec = max(0.0, self.consecutive_speech_sec - dt * 1.5)

        # 4. Acoustic Environment Classification
        if rms < self.silence_threshold:
            env = "SILENT_WORK"
            chatter = 0.05
            engagement_factor = 0.70  # Calm study/writing
        elif is_speech and self.consecutive_speech_sec > 12.0:
            env = "INSTRUCTOR_MONOLOGUE"
            chatter = 0.10
            engagement_factor = 0.65  # Passive listening
        elif is_speech and rms < self.chatter_threshold:
            env = "CLASSROOM_DISCUSSION"
            chatter = 0.20
            engagement_factor = 0.95  # Active Q&A participation
        else:
            env = "SIDE_CHATTER"
            chatter = min(1.0, (rms - self.silence_threshold) / 0.15)
            engagement_factor = 0.40  # Distracted noise

        with self.lock:
            self.current_state = AudioAcousticState(
                rms_energy=round(rms, 4),
                zero_crossing_rate=round(zcr, 4),
                is_speech_active=is_speech,
                acoustic_environment=env,
                speech_duration_sec=round(self.consecutive_speech_sec, 1),
                chatter_index=round(chatter, 2),
                audio_engagement_factor=round(engagement_factor, 2),
            )

    def _simulated_audio_loop(self):
        """Simulates realistic classroom audio shifts when hardware mic is absent."""
        phase = 0
        while self.is_running:
            time.sleep(1.0)
            phase = (phase + 1) % 60
            # Alternate between discussion, active lecture, and quiet periods
            if phase < 25:
                # Lecture
                synth_rms = 0.045 + np.random.uniform(-0.005, 0.005)
                synth_audio = np.random.normal(0, synth_rms, 8000)
            elif phase < 45:
                # Active Discussion
                synth_rms = 0.065 + np.random.uniform(-0.010, 0.015)
                synth_audio = np.random.normal(0, synth_rms, 8000)
            else:
                # Quiet focus
                synth_rms = 0.010 + np.random.uniform(-0.002, 0.002)
                synth_audio = np.random.normal(0, synth_rms, 8000)

            self._process_chunk(synth_audio)

    def get_state(self) -> AudioAcousticState:
        with self.lock:
            return self.current_state

    def stop(self):
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
