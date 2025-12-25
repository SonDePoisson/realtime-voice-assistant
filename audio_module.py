import asyncio
import logging
import os
import struct
import threading
import time
from collections import namedtuple
from queue import Queue
from typing import Callable, Generator, Optional

import numpy as np
from huggingface_hub import hf_hub_download

# Assuming RealtimeTTS is installed and available
from RealtimeTTS import EdgeEngine, TextToAudioStream

logger = logging.getLogger(__name__)

# Default configuration constants
EDGE_TTS_VOICE = "en-US-AvaMultilingualNeural"  # Voix multilingue
START_ENGINE = "edgeTTS"
Silence = namedtuple("Silence", ("comma", "sentence", "default"))
ENGINE_SILENCES = {
    "edgeTTS": Silence(comma=0.3, sentence=0.6, default=0.3),
}
# Stream chunk sizes influence latency vs. throughput trade-offs
QUICK_ANSWER_STREAM_CHUNK_SIZE = 8
FINAL_ANSWER_STREAM_CHUNK_SIZE = 30


# Coqui model download helper functions
def create_directory(path: str) -> None:
    """
    Creates a directory at the specified path if it doesn't already exist.

    Args:
        path: The directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)


class AudioProcessor:
    """
    Manages Text-to-Speech (TTS) synthesis using various engines via RealtimeTTS.

    This class initializes a chosen TTS engine (Coqui, Kokoro, or Orpheus),
    configures it for streaming output, measures initial latency (TTFT),
    and provides methods to synthesize audio from text strings or generators,
    placing the resulting audio chunks into a queue. It handles dynamic
    stream parameter adjustments and manages the synthesis lifecycle, including
    optional callbacks upon receiving the first audio chunk.
    """

    def __init__(
        self,
        engine: str = START_ENGINE,
    ) -> None:
        """
        Initializes the AudioProcessor with a specific TTS engine.

        Sets up the chosen engine (edgeTTS)
        if necessary, configures the RealtimeTTS stream, and performs an initial
        synthesis to measure Time To First Audio chunk (TTFA).

        Args:
            engine: The name of the TTS engine to use ("EdgeTTS").
        """
        self.engine_name = engine
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.audio_chunks = asyncio.Queue()  # Queue for synthesized audio output

        self.silence = ENGINE_SILENCES.get(engine, ENGINE_SILENCES[self.engine_name])
        self.current_stream_chunk_size = QUICK_ANSWER_STREAM_CHUNK_SIZE  # Initial chunk size

        # Dynamically load and configure the selected TTS engine
        if engine == "edgeTTS":
            self.engine = EdgeEngine()
            self.engine.set_voice(EDGE_TTS_VOICE)
        else:
            raise ValueError(f"Unsupported engine: {engine}")

        # Initialize the RealtimeTTS stream
        self.stream = TextToAudioStream(
            self.engine,
            muted=False,  # Do not play audio directly
            playout_chunk_size=4096,  # Internal chunk size for processing
            on_audio_stream_stop=self.on_audio_stream_stop,
        )

        # Prewarm the engine
        self.stream.feed("prewarm")
        play_kwargs = dict(
            log_synthesized_text=False,  # Don't log prewarm text
            muted=True,
            fast_sentence_fragment=False,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999,  # Effectively disable this
        )
        self.stream.play(**play_kwargs)  # Synchronous play for prewarm
        # Wait for prewarm to finish (indicated by on_audio_stream_stop)
        while self.stream.is_playing():
            time.sleep(0.01)
        self.finished_event.wait()  # Wait for stop callback
        self.finished_event.clear()

        # Measure Time To First Audio (TTFA)
        start_time = time.time()
        ttfa = None

        def on_audio_chunk_ttfa(chunk: bytes):
            nonlocal ttfa
            if ttfa is None:
                ttfa = time.time() - start_time
                logger.debug(f"👄⏱️ TTFA measurement first chunk arrived, TTFA: {ttfa:.2f}s.")

        self.stream.feed("This is a test sentence to measure the time to first audio chunk.")
        play_kwargs_ttfa = dict(
            on_audio_chunk=on_audio_chunk_ttfa,
            log_synthesized_text=False,  # Don't log test sentence
            muted=True,
            fast_sentence_fragment=False,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999,
        )
        self.stream.play_async(**play_kwargs_ttfa)

        # Wait until the first chunk arrives or stream finishes
        while ttfa is None and (self.stream.is_playing() or not self.finished_event.is_set()):
            time.sleep(0.01)
        self.stream.stop()  # Ensure stream stops cleanly

        # Wait for stop callback if it hasn't fired yet
        if not self.finished_event.is_set():
            self.finished_event.wait(timeout=2.0)  # Add timeout for safety
        self.finished_event.clear()

        if ttfa is not None:
            logger.debug(f"👄⏱️ TTFA measurement complete. TTFA: {ttfa:.2f}s.")
            self.tts_inference_time = ttfa * 1000  # Store as ms
        else:
            logger.warning("👄⚠️ TTFA measurement failed (no audio chunk received).")
            self.tts_inference_time = 0

        # Callbacks to be set externally if needed
        self.on_first_audio_chunk_synthesize: Optional[Callable[[], None]] = None

    def on_audio_stream_stop(self) -> None:
        """
        Callback executed when the RealtimeTTS audio stream stops processing.

        Logs the event and sets the `finished_event` to signal completion or stop.
        """
        logger.info("👄🛑 Audio stream stopped.")
        self.finished_event.set()

    def synthesize(
        self,
        text: str,
        audio_chunks: Queue,
        stop_event: threading.Event,
        generation_string: str = "",
    ) -> bool:
        """
        Synthesizes audio from a complete text string and puts chunks into a queue.

        Feeds the entire text string to the TTS engine. As audio chunks are generated,
        they are potentially buffered initially for smoother streaming and then put
        into the provided queue. Synthesis can be interrupted via the stop_event. Triggers the
        `on_first_audio_chunk_synthesize` callback when the first valid audio chunk is queued.

        Args:
            text: The text string to synthesize.
            audio_chunks: The queue to put the resulting audio chunks (bytes) into.
                          This should typically be the instance's `self.audio_chunks`.
            stop_event: A threading.Event to signal interruption of the synthesis.
                        This should typically be the instance's `self.stop_event`.
            generation_string: An optional identifier string for logging purposes.

        Returns:
            True if synthesis completed fully, False if interrupted by stop_event.
        """
        self.stream.feed(text)
        self.finished_event.clear()  # Reset finished event before starting

        # Buffering state variables
        buffer: list[bytes] = []
        good_streak: int = 0
        buffering: bool = True
        buf_dur: float = 0.0
        SR, BPS = 24000, 2  # Assumed Sample Rate and Bytes Per Sample (16-bit)
        start = time.time()
        self._quick_prev_chunk_time: float = 0.0  # Track time of previous chunk

        def on_audio_chunk(chunk: bytes):
            nonlocal buffer, good_streak, buffering, buf_dur, start
            # Check for interruption signal
            if stop_event.is_set():
                logger.info(
                    f"👄🛑 {generation_string} Quick audio stream interrupted by stop_event. Text: {text[:50]}..."
                )
                # We should not put more chunks, let the main loop handle stream stop
                return

            now = time.time()
            samples = len(chunk) // BPS
            play_duration = samples / SR  # Duration of the current chunk

            # --- Timing and Logging ---
            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                self._quick_prev_chunk_time = now
                ttfa_actual = now - start
                logger.info(
                    f"👄🚀 {generation_string} Quick audio start. TTFA: {ttfa_actual:.2f}s. Text: {text[:50]}..."
                )
            else:
                gap = now - self._quick_prev_chunk_time
                self._quick_prev_chunk_time = now
                if gap <= play_duration * 1.1:  # Allow small tolerance
                    # logger.debug(f"👄✅ {generation_string} Quick chunk ok (gap={gap:.3f}s ≤ {play_duration:.3f}s). Text: {text[:50]}...")
                    good_streak += 1
                else:
                    logger.warning(
                        f"👄❌ {generation_string} Quick chunk slow (gap={gap:.3f}s > {play_duration:.3f}s). Text: {text[:50]}..."
                    )
                    good_streak = 0  # Reset streak on slow chunk

            put_occurred_this_call = False  # Track if put happened in this specific call

            # --- Buffering Logic ---
            buffer.append(chunk)  # Always append the received chunk first
            buf_dur += play_duration  # Update buffer duration

            if buffering:
                # Check conditions to flush buffer and stop buffering
                if good_streak >= 2 or buf_dur >= 0.5:  # Flush if stable or buffer > 0.5s
                    logger.info(
                        f"👄➡️ {generation_string} Quick Flushing buffer (streak={good_streak}, dur={buf_dur:.2f}s)."
                    )
                    for c in buffer:
                        try:
                            audio_chunks.put_nowait(c)
                            put_occurred_this_call = True
                        except asyncio.QueueFull:
                            logger.warning(f"👄⚠️ {generation_string} Quick audio queue full, dropping chunk.")
                    buffer.clear()
                    buf_dur = 0.0  # Reset buffer duration
                    buffering = False  # Stop buffering mode
            else:  # Not buffering, put chunk directly
                try:
                    audio_chunks.put_nowait(chunk)
                    put_occurred_this_call = True
                except asyncio.QueueFull:
                    logger.warning(f"👄⚠️ {generation_string} Quick audio queue full, dropping chunk.")

            # --- First Chunk Callback ---
            if put_occurred_this_call and not on_audio_chunk.callback_fired:
                if self.on_first_audio_chunk_synthesize:
                    try:
                        logger.info(f"👄🚀 {generation_string} Quick Firing on_first_audio_chunk_synthesize.")
                        self.on_first_audio_chunk_synthesize()
                    except Exception as e:
                        logger.error(
                            f"👄💥 {generation_string} Quick Error in on_first_audio_chunk_synthesize callback: {e}",
                            exc_info=True,
                        )
                # Ensure callback fires only once per synthesize call
                on_audio_chunk.callback_fired = True

        # Initialize callback state for this run
        on_audio_chunk.first_call = True
        on_audio_chunk.callback_fired = False

        play_kwargs = dict(
            log_synthesized_text=True,  # Log the text being synthesized
            on_audio_chunk=on_audio_chunk,
            muted=False,  # We handle audio via the queue
            fast_sentence_fragment=False,  # Standard processing
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999,  # Don't force early fragments
        )

        logger.info(f"👄▶️ {generation_string} Quick Starting synthesis. Text: {text[:50]}...")
        self.stream.play_async(**play_kwargs)

        # Wait loop for completion or interruption
        while self.stream.is_playing() or not self.finished_event.is_set():
            if stop_event.is_set():
                self.stream.stop()
                logger.info(
                    f"👄🛑 {generation_string} Quick answer synthesis aborted by stop_event. Text: {text[:50]}..."
                )
                # Drain remaining buffer if any? Decided against it to stop faster.
                buffer.clear()
                # Wait briefly for stop confirmation? The finished_event handles this.
                self.finished_event.wait(timeout=1.0)  # Wait for stream stop confirmation
                return False  # Indicate interruption
            time.sleep(0.01)

        # # If loop exited normally, check if buffer still has content (stream finished before flush)
        if buffering and buffer and not stop_event.is_set():
            logger.info(f"👄➡️ {generation_string} Quick Flushing remaining buffer after stream finished.")
            for c in buffer:
                try:
                    audio_chunks.put_nowait(c)
                except asyncio.QueueFull:
                    logger.warning(f"👄⚠️ {generation_string} Quick audio queue full on final flush, dropping chunk.")
            buffer.clear()

        logger.info(f"👄✅ {generation_string} Quick answer synthesis complete. Text: {text[:50]}...")
        return True  # Indicate successful completion

    def synthesize_generator(
        self,
        generator: Generator[str, None, None],
        audio_chunks: Queue,  # Should match self.audio_chunks type
        stop_event: threading.Event,
        generation_string: str = "",
    ) -> bool:
        """
         Synthesizes audio from a generator yielding text chunks and puts audio into a queue.

         Feeds text chunks yielded by the generator to the TTS engine. As audio chunks
         are generated, they are potentially buffered initially and then put into the
         provided queue. Synthesis can be interrupted via the stop_event. Triggers the
        `on_first_audio_chunk_synthesize` callback when the first valid audio chunk is queued.


         Args:
             generator: A generator yielding text chunks (strings) to synthesize.
             audio_chunks: The queue to put the resulting audio chunks (bytes) into.
                           This should typically be the instance's `self.audio_chunks`.
             stop_event: A threading.Event to signal interruption of the synthesis.
                         This should typically be the instance's `self.stop_event`.
             generation_string: An optional identifier string for logging purposes.

         Returns:
             True if synthesis completed fully, False if interrupted by stop_event.
        """

        # Feed the generator to the stream
        self.stream.feed(generator)
        self.finished_event.clear()  # Reset finished event

        # Buffering state variables
        buffer: list[bytes] = []
        good_streak: int = 0
        buffering: bool = True
        buf_dur: float = 0.0
        SR, BPS = 24000, 2  # Assumed Sample Rate and Bytes Per Sample
        start = time.time()
        self._final_prev_chunk_time: float = 0.0  # Separate timer for generator synthesis

        def on_audio_chunk(chunk: bytes):
            nonlocal buffer, good_streak, buffering, buf_dur, start
            if stop_event.is_set():
                logger.info(f"👄🛑 {generation_string} Final audio stream interrupted by stop_event.")
                return

            now = time.time()
            samples = len(chunk) // BPS
            play_duration = samples / SR

            # --- Timing and Logging ---
            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                self._final_prev_chunk_time = now
                ttfa_actual = now - start
                logger.info(f"👄🚀 {generation_string} Final audio start. TTFA: {ttfa_actual:.2f}s.")
            else:
                gap = now - self._final_prev_chunk_time
                self._final_prev_chunk_time = now
                if gap <= play_duration * 1.1:
                    # logger.debug(f"👄✅ {generation_string} Final chunk ok (gap={gap:.3f}s ≤ {play_duration:.3f}s).")
                    good_streak += 1
                else:
                    logger.warning(
                        f"👄❌ {generation_string} Final chunk slow (gap={gap:.3f}s > {play_duration:.3f}s)."
                    )
                    good_streak = 0

            put_occurred_this_call = False

            # --- Buffering Logic ---
            buffer.append(chunk)
            buf_dur += play_duration
            if buffering:
                if good_streak >= 2 or buf_dur >= 0.5:  # Same flush logic as synthesize
                    logger.info(
                        f"👄➡️ {generation_string} Final Flushing buffer (streak={good_streak}, dur={buf_dur:.2f}s)."
                    )
                    for c in buffer:
                        try:
                            audio_chunks.put_nowait(c)
                            put_occurred_this_call = True
                        except asyncio.QueueFull:
                            logger.warning(f"👄⚠️ {generation_string} Final audio queue full, dropping chunk.")
                    buffer.clear()
                    buf_dur = 0.0
                    buffering = False
            else:  # Not buffering
                try:
                    audio_chunks.put_nowait(chunk)
                    put_occurred_this_call = True
                except asyncio.QueueFull:
                    logger.warning(f"👄⚠️ {generation_string} Final audio queue full, dropping chunk.")

            # --- First Chunk Callback --- (Using the same callback as synthesize)
            if put_occurred_this_call and not on_audio_chunk.callback_fired:
                if self.on_first_audio_chunk_synthesize:
                    try:
                        logger.info(f"👄🚀 {generation_string} Final Firing on_first_audio_chunk_synthesize.")
                        self.on_first_audio_chunk_synthesize()
                    except Exception as e:
                        logger.error(
                            f"👄💥 {generation_string} Final Error in on_first_audio_chunk_synthesize callback: {e}",
                            exc_info=True,
                        )
                on_audio_chunk.callback_fired = True

        # Initialize callback state
        on_audio_chunk.first_call = True
        on_audio_chunk.callback_fired = False

        play_kwargs = dict(
            log_synthesized_text=True,  # Log text from generator
            on_audio_chunk=on_audio_chunk,
            muted=False,
            fast_sentence_fragment=False,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999,
        )

        logger.info(f"👄▶️ {generation_string} Final Starting synthesis from generator.")
        self.stream.play_async(**play_kwargs)

        # Wait loop for completion or interruption
        while self.stream.is_playing() or not self.finished_event.is_set():
            if stop_event.is_set():
                self.stream.stop()
                logger.info(f"👄🛑 {generation_string} Final answer synthesis aborted by stop_event.")
                buffer.clear()
                self.finished_event.wait(timeout=1.0)  # Wait for stream stop confirmation
                return False  # Indicate interruption
            time.sleep(0.01)

        # Flush remaining buffer if stream finished before flush condition met
        if buffering and buffer and not stop_event.is_set():
            logger.info(f"👄➡️ {generation_string} Final Flushing remaining buffer after stream finished.")
            for c in buffer:
                try:
                    audio_chunks.put_nowait(c)
                except asyncio.QueueFull:
                    logger.warning(f"👄⚠️ {generation_string} Final audio queue full on final flush, dropping chunk.")
            buffer.clear()

        logger.info(f"👄✅ {generation_string} Final answer synthesis complete.")
        return True  # Indicate successful completion
