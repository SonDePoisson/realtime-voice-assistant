import asyncio
import logging
import threading
import time
from collections import namedtuple
from queue import Queue
from typing import Callable, Generator, Optional

import edge_tts

# Assuming RealtimeTTS is installed and available
from RealtimeTTS import (
    KokoroEngine,
    TextToAudioStream,
)

# Supprimer les warnings de phonemizer
logging.getLogger("phonemizer").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Default configuration constants
START_ENGINE = "edge_tts"
KOKORO_VOICES = {
    "fr": "ff_siwis",  # Voix française féminine
    "en": "af_heart",  # Voix anglaise féminine
}
EDGE_TTS_VOICES = {
    "fr": "fr-FR-DeniseNeural",  # Voix française féminine
    "en": "en-US-AvaMultilingualNeural",  # Voix anglaise multilingue
}
Silence = namedtuple("Silence", ("comma", "sentence", "default"))
ENGINE_SILENCES = {
    "kokoro": Silence(comma=0.3, sentence=0.6, default=0.3),
    "edge_tts": Silence(comma=0.3, sentence=0.6, default=0.3),
}
# Stream chunk sizes influence latency vs. throughput trade-offs (used by Kokoro)
QUICK_ANSWER_STREAM_CHUNK_SIZE = 8
FINAL_ANSWER_STREAM_CHUNK_SIZE = 30


class AudioProcessor:
    """
    Manages Text-to-Speech (TTS) synthesis using various engines.

    This class initializes a chosen TTS engine (Kokoro or EdgeTTS),
    configures it for streaming output, measures initial latency (TTFT),
    and provides methods to synthesize audio from text strings or generators,
    placing the resulting audio chunks into a queue. It handles dynamic
    stream parameter adjustments and manages the synthesis lifecycle, including
    optional callbacks upon receiving the first audio chunk.
    """

    def __init__(
        self,
        engine: str = START_ENGINE,
        language: str = "fr",
    ) -> None:
        """
        Initializes the AudioProcessor with a specific TTS engine.

        Sets up the chosen engine (Kokoro or EdgeTTS), configures the stream,
        and performs an initial synthesis to measure Time To First Audio chunk (TTFA).

        Args:
            engine: The name of the TTS engine to use ("kokoro", "edge_tts").
            language: Language code for voice selection (e.g., "fr", "en").
        """
        self.engine_name = engine
        self.language = language
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.audio_chunks = asyncio.Queue()  # Queue for synthesized audio output

        self.silence = ENGINE_SILENCES.get(engine, ENGINE_SILENCES[self.engine_name])
        self.current_stream_chunk_size = QUICK_ANSWER_STREAM_CHUNK_SIZE  # Initial chunk size

        # Dynamically load and configure the selected TTS engine
        if engine == "kokoro":
            # Sélectionner la voix selon la langue
            voice = KOKORO_VOICES.get(language, KOKORO_VOICES["fr"])
            logger.debug(f"Using Kokoro voice '{voice}' for language '{language}'")

            self.engine = KokoroEngine(
                voice=voice,
                default_speed=1.26,
                trim_silence=True,
                silence_threshold=0.01,
                extra_start_ms=25,
                extra_end_ms=15,
                fade_in_ms=15,
                fade_out_ms=10,
            )
        elif engine == "edge_tts":
            # EdgeTTS ne nécessite pas d'initialisation d'engine
            # La synthèse se fait directement via edge_tts.Communicate()
            self.edge_voice = EDGE_TTS_VOICES.get(language, EDGE_TTS_VOICES["en"])
            logger.debug(f"Using EdgeTTS voice '{self.edge_voice}' for language '{language}'")
            self.engine = None  # EdgeTTS n'utilise pas RealtimeTTS
            self.stream = None  # Pas de stream pour EdgeTTS
        else:
            raise ValueError(f"Unsupported engine: {engine}")

        # Initialize the RealtimeTTS stream (only for Kokoro)
        if engine == "kokoro":
            self.stream = TextToAudioStream(
                self.engine,
                muted=False,  # CHANGÉ: False pour lecture directe sur haut-parleurs
                playout_chunk_size=4096,  # Internal chunk size for processing
                on_audio_stream_stop=self.on_audio_stream_stop,
            )

        # Prewarm and measure TTFA (only for Kokoro)
        if engine == "kokoro":
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
                    logger.debug(f"️ TTFA measurement first chunk arrived, TTFA: {ttfa:.2f}s.")

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
                logger.debug(f"️ TTFA measurement complete. TTFA: {ttfa:.2f}s.")
                self.tts_inference_time = ttfa * 1000  # Store as ms
            else:
                logger.warning("️ TTFA measurement failed (no audio chunk received).")
                self.tts_inference_time = 0
        else:
            # For EdgeTTS, skip prewarm and set a default TTFA
            self.tts_inference_time = 200  # Estimated TTFA for EdgeTTS

        # Callbacks to be set externally if needed
        self.on_first_audio_chunk_synthesize: Optional[Callable[[], None]] = None

    def on_audio_stream_stop(self) -> None:
        """
        Callback executed when the RealtimeTTS audio stream stops processing.

        Logs the event and sets the `finished_event` to signal completion or stop.
        """
        logger.debug("Audio stream stopped.")
        self.finished_event.set()

    async def _synthesize_edge_tts(
        self,
        text: str,
        audio_chunks: Queue,
        stop_event: threading.Event,
    ) -> bool:
        """
        Synthétise le texte avec EdgeTTS de manière asynchrone.

        Args:
            text: Le texte à synthétiser
            audio_chunks: Queue pour les chunks audio
            stop_event: Event pour arrêter la synthèse

        Returns:
            True si la synthèse est complète, False si interrompue
        """
        try:
            communicate = edge_tts.Communicate(text, self.edge_voice)
            first_chunk = True

            async for chunk in communicate.stream():
                if stop_event.is_set():
                    logger.debug("EdgeTTS synthesis interrupted")
                    return False

                if chunk["type"] == "audio":
                    audio_data = chunk["data"]
                    try:
                        audio_chunks.put_nowait(audio_data)

                        # Appeler le callback au premier chunk
                        if first_chunk and self.on_first_audio_chunk_synthesize:
                            first_chunk = False
                            try:
                                self.on_first_audio_chunk_synthesize()
                            except Exception as e:
                                logger.error(f"Error in on_first_audio_chunk_synthesize callback: {e}")
                    except Exception as e:
                        logger.warning(f"EdgeTTS audio queue full, dropping chunk: {e}")

            return True
        except Exception as e:
            logger.error(f"EdgeTTS synthesis error: {e}", exc_info=True)
            return False

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
        into the provided queue. Synthesis can be interrupted via the stop_event.
        Triggers the `on_first_audio_chunk_synthesize` callback when the first valid audio chunk is queued.

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
        # Handle EdgeTTS separately (async)
        if self.engine_name == "edge_tts":
            logger.debug(f"▶️ {generation_string} Starting EdgeTTS synthesis. Text: {text[:50]}...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._synthesize_edge_tts(text, audio_chunks, stop_event)
                )
                if result:
                    logger.debug(f"{generation_string} EdgeTTS synthesis complete.")
                return result
            finally:
                loop.close()

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
                logger.debug(f" {generation_string} Quick audio stream interrupted by stop_event. Text: {text[:50]}...")
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
                logger.debug(f" {generation_string} Quick audio start. TTFA: {ttfa_actual:.2f}s. Text: {text[:50]}...")
            else:
                gap = now - self._quick_prev_chunk_time
                self._quick_prev_chunk_time = now
                if gap <= play_duration * 1.1:  # Allow small tolerance
                    # logger.debug(f"{generation_string} Quick chunk ok (gap={gap:.3f}s ≤ {play_duration:.3f}s). Text: {text[:50]}...")
                    good_streak += 1
                else:
                    logger.debug(
                        f" {generation_string} Quick chunk slow (gap={gap:.3f}s > {play_duration:.3f}s). Text: {text[:50]}..."
                    )
                    good_streak = 0  # Reset streak on slow chunk

            put_occurred_this_call = False  # Track if put happened in this specific call

            # --- Buffering Logic ---
            buffer.append(chunk)  # Always append the received chunk first
            buf_dur += play_duration  # Update buffer duration

            if buffering:
                # Check conditions to flush buffer and stop buffering
                if good_streak >= 2 or buf_dur >= 0.5:  # Flush if stable or buffer > 0.5s
                    logger.debug(
                        f"️ {generation_string} Quick Flushing buffer (streak={good_streak}, dur={buf_dur:.2f}s)."
                    )
                    for c in buffer:
                        try:
                            audio_chunks.put_nowait(c)
                            put_occurred_this_call = True
                        except asyncio.QueueFull:
                            logger.warning(f"️ {generation_string} Quick audio queue full, dropping chunk.")
                    buffer.clear()
                    buf_dur = 0.0  # Reset buffer duration
                    buffering = False  # Stop buffering mode
            else:  # Not buffering, put chunk directly
                try:
                    audio_chunks.put_nowait(chunk)
                    put_occurred_this_call = True
                except asyncio.QueueFull:
                    logger.warning(f"️ {generation_string} Quick audio queue full, dropping chunk.")

            # --- First Chunk Callback ---
            if put_occurred_this_call and not on_audio_chunk.callback_fired:
                if self.on_first_audio_chunk_synthesize:
                    try:
                        logger.debug(f"{generation_string} Quick Firing on_first_audio_chunk_synthesize.")
                        self.on_first_audio_chunk_synthesize()
                    except Exception as e:
                        logger.error(
                            f" {generation_string} Quick Error in on_first_audio_chunk_synthesize callback: {e}",
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
            muted=True,  # We handle audio via the queue
            fast_sentence_fragment=False,  # Standard processing
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999,  # Don't force early fragments
        )

        logger.debug(f"▶️ {generation_string} Quick Starting synthesis. Text: {text[:50]}...")
        self.stream.play_async(**play_kwargs)

        # Wait loop for completion or interruption
        while self.stream.is_playing() or not self.finished_event.is_set():
            if stop_event.is_set():
                self.stream.stop()
                logger.debug(f" {generation_string} Quick answer synthesis aborted by stop_event. Text: {text[:50]}...")
                # Drain remaining buffer if any? Decided against it to stop faster.
                buffer.clear()
                # Wait briefly for stop confirmation? The finished_event handles this.
                self.finished_event.wait(timeout=1.0)  # Wait for stream stop confirmation
                return False  # Indicate interruption
            time.sleep(0.01)

        # # If loop exited normally, check if buffer still has content (stream finished before flush)
        if buffering and buffer and not stop_event.is_set():
            logger.debug(f"️ {generation_string} Quick Flushing remaining buffer after stream finished.")
            for c in buffer:
                try:
                    audio_chunks.put_nowait(c)
                except asyncio.QueueFull:
                    logger.warning(f"️ {generation_string} Quick audio queue full on final flush, dropping chunk.")
            buffer.clear()

        logger.debug(f"{generation_string} Quick answer synthesis complete. Text: {text[:50]}...")
        return True  # Indicate successful completion

    def synthesize_generator(
        self,
        generator: Generator[str, None, None],
        audio_chunks: Optional[Queue] = None,
        stop_event: Optional[threading.Event] = None,
        generation_string: str = "",
    ) -> bool:
        """
         Synthesizes audio from a generator yielding text chunks.

         Feeds text chunks yielded by the generator to the TTS engine. If audio_chunks
         is provided, audio is put into the queue (muted playback). If audio_chunks is None,
         audio is played directly to speakers. Synthesis can be interrupted via the stop_event.
         Triggers the `on_first_audio_chunk_synthesize` callback when the first valid audio chunk is queued.


         Args:
             generator: A generator yielding text chunks (strings) to synthesize.
             audio_chunks: Optional queue to put the resulting audio chunks (bytes) into.
                           If None, audio is played directly to speakers.
             stop_event: A threading.Event to signal interruption of the synthesis.
                         If None, uses self.stop_event.
             generation_string: An optional identifier string for logging purposes.

         Returns:
             True if synthesis completed fully, False if interrupted by stop_event.
        """
        # Use default stop_event if not provided
        if stop_event is None:
            stop_event = self.stop_event

        # Handle EdgeTTS (accumulate generator into full text)
        if self.engine_name == "edge_tts":
            # EdgeTTS doesn't support streaming from generator, so accumulate text first
            full_text = "".join(generator)
            if audio_chunks is None:
                # If no queue provided, we need one for EdgeTTS
                logger.warning("EdgeTTS requires audio_chunks queue, cannot play directly")
                return False
            return self.synthesize(full_text, audio_chunks, stop_event, generation_string)

        # Determine if we're using queue mode or direct playback
        use_queue = audio_chunks is not None

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
                logger.debug(f"{generation_string} Final audio stream interrupted by stop_event.")
                return

            now = time.time()
            samples = len(chunk) // BPS
            play_duration = samples / SR

            # --- Timing and Logging ---
            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                self._final_prev_chunk_time = now
                ttfa_actual = now - start
                logger.debug(f"{generation_string} Final audio start. TTFA: {ttfa_actual:.2f}s.")
            else:
                gap = now - self._final_prev_chunk_time
                self._final_prev_chunk_time = now
                if gap <= play_duration * 1.1:
                    # logger.debug(f"{generation_string} Final chunk ok (gap={gap:.3f}s ≤ {play_duration:.3f}s).")
                    good_streak += 1
                else:
                    logger.debug(f" {generation_string} Final chunk slow (gap={gap:.3f}s > {play_duration:.3f}s).")
                    good_streak = 0

            put_occurred_this_call = False

            # --- Buffering Logic (only if using queue) ---
            if use_queue:
                buffer.append(chunk)
                buf_dur += play_duration
                if buffering:
                    if good_streak >= 2 or buf_dur >= 0.5:  # Same flush logic as synthesize
                        logger.debug(
                            f"️ {generation_string} Final Flushing buffer (streak={good_streak}, dur={buf_dur:.2f}s)."
                        )
                        for c in buffer:
                            try:
                                audio_chunks.put_nowait(c)
                                put_occurred_this_call = True
                            except asyncio.QueueFull:
                                logger.warning(f"️ {generation_string} Final audio queue full, dropping chunk.")
                        buffer.clear()
                        buf_dur = 0.0
                        buffering = False
                else:  # Not buffering
                    try:
                        audio_chunks.put_nowait(chunk)
                        put_occurred_this_call = True
                    except asyncio.QueueFull:
                        logger.warning(f"️ {generation_string} Final audio queue full, dropping chunk.")
            else:
                # Direct playback mode - just mark that we got audio
                put_occurred_this_call = True

            # --- First Chunk Callback --- (Using the same callback as synthesize)
            if put_occurred_this_call and not on_audio_chunk.callback_fired:
                if self.on_first_audio_chunk_synthesize:
                    try:
                        logger.debug(f"{generation_string} Final Firing on_first_audio_chunk_synthesize.")
                        self.on_first_audio_chunk_synthesize()
                    except Exception as e:
                        logger.error(
                            f" {generation_string} Final Error in on_first_audio_chunk_synthesize callback: {e}",
                            exc_info=True,
                        )
                on_audio_chunk.callback_fired = True

        # Initialize callback state
        on_audio_chunk.first_call = True
        on_audio_chunk.callback_fired = False

        play_kwargs = dict(
            log_synthesized_text=True,  # Log text from generator
            on_audio_chunk=on_audio_chunk if use_queue else None,
            muted=use_queue,  # Muted if using queue, direct playback otherwise
            fast_sentence_fragment=False,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999,
        )

        # Add Orpheus specific parameters for generator streaming
        if self.engine_name == "orpheus":
            # These encourage waiting for more text before synthesizing, potentially better for generators
            play_kwargs["minimum_sentence_length"] = 200
            play_kwargs["minimum_first_fragment_length"] = 200

        logger.debug(f"▶️ {generation_string} Final Starting synthesis from generator.")
        self.stream.play_async(**play_kwargs)

        # Wait loop for completion or interruption
        while self.stream.is_playing() or not self.finished_event.is_set():
            if stop_event.is_set():
                self.stream.stop()
                logger.debug(f"{generation_string} Final answer synthesis aborted by stop_event.")
                buffer.clear()
                self.finished_event.wait(timeout=1.0)  # Wait for stream stop confirmation
                return False  # Indicate interruption
            time.sleep(0.01)

        # Flush remaining buffer if stream finished before flush condition met
        if buffering and buffer and not stop_event.is_set():
            logger.debug(f"️ {generation_string} Final Flushing remaining buffer after stream finished.")
            for c in buffer:
                try:
                    audio_chunks.put_nowait(c)
                except asyncio.QueueFull:
                    logger.warning(f"️ {generation_string} Final audio queue full on final flush, dropping chunk.")
            buffer.clear()

        logger.debug(f"{generation_string} Final answer synthesis complete.")
        return True  # Indicate successful completion
