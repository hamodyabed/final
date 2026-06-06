"""Microphone recorder that writes WAV files using sounddevice + soundfile.

Designed to be driven from the Qt main thread:
    rec = AudioRecorder()
    rec.start(Path("answer_1.wav"))
    ...
    rec.stop()
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    """Background recorder that streams microphone audio to a WAV file."""

    def __init__(self, samplerate: int = 44_100, channels: int = 1) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._output_path: Optional[Path] = None
        self._error: Optional[BaseException] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def start(self, output_path: Path) -> None:
        """Begin recording to ``output_path`` (.wav)."""
        if self.is_recording:
            raise RuntimeError("Recorder is already active.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path = output_path
        self._stop_event.clear()
        self._error = None
        # Drain anything left over from a previous session.
        with self._queue.mutex:
            self._queue.queue.clear()

        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="audio-writer", daemon=True
        )
        self._writer_thread.start()

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="int16",
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> Optional[Path]:
        """Stop recording and return the saved file path (or ``None``).

        We deliberately:
          1. Stop *and close* the PortAudio stream first (no more callbacks
             will fire after ``close``).
          2. Then signal the writer thread to drain the rest of the queue,
             so the last audio blocks that were already enqueued by the
             final callback get flushed to disk.
          3. Finally join the writer thread without a tight timeout so the
             trailing audio is never truncated.
        """
        if not self.is_recording:
            return None

        assert self._stream is not None
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

        # Give PortAudio a moment to deliver any pending callback to the queue
        # before we tell the writer it is allowed to exit.
        import time
        time.sleep(0.05)

        self._stop_event.set()
        if self._writer_thread is not None:
            # Long enough for the writer to drain the queue even on slow disks,
            # short enough that a hang doesn't lock the UI forever.
            self._writer_thread.join(timeout=15)
            self._writer_thread = None

        if self._error is not None:
            raise self._error  # surface I/O errors to the caller
        return self._output_path

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status) -> None:  # noqa: D401
        """Called by sounddevice on its own thread for every audio block."""
        if status:
            # Underflows / overflows are non-fatal; just ignore them.
            pass
        # ``indata`` is reused by PortAudio, so we must copy.
        self._queue.put(indata.copy())

    def _writer_loop(self) -> None:
        """Pull blocks off the queue and append them to the WAV file."""
        if self._output_path is None:
            return
        try:
            with sf.SoundFile(
                str(self._output_path),
                mode="w",
                samplerate=self.samplerate,
                channels=self.channels,
                subtype="PCM_16",
            ) as sink:
                while not (self._stop_event.is_set() and self._queue.empty()):
                    try:
                        block = self._queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    sink.write(block)
        except BaseException as exc:  # noqa: BLE001 – re-raised from stop()
            self._error = exc
