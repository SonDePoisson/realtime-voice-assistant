import logging
import threading
import time
from typing import Callable, Generator, Optional

from RealtimeTTS import EdgeEngine, TextToAudioStream

logger = logging.getLogger(__name__)

# Configuration EdgeTTS
EDGE_TTS_VOICE = "en-US-AvaMultilingualNeural"  # Voix multilingue
SILENCE_DURATIONS = {
    "comma": 0.3,
    "sentence": 0.6,
    "default": 0.3,
}


class AudioProcessor:
    """
    Manages Text-to-Speech (TTS) synthesis using EdgeTTS.

    EdgeTTS generates MP3/Opus audio which is played directly via MPV
    (no PCM queue needed).
    """

    def __init__(self) -> None:
        """
        Initializes the AudioProcessor with EdgeTTS engine.

        Sets up EdgeTTS, configures the stream, and performs an initial
        synthesis to measure Time To First Audio chunk (TTFA).
        """
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()

        # Initialize EdgeTTS engine
        logger.debug(f"Initializing EdgeTTS with voice '{EDGE_TTS_VOICE}'")
        self.engine = EdgeEngine()
        self.engine.set_voice(EDGE_TTS_VOICE)

        # Initialize the RealtimeTTS stream
        self.stream = TextToAudioStream(
            self.engine,
            muted=False,  # Direct playback via MPV
            playout_chunk_size=4096,
            on_audio_stream_stop=self.on_audio_stream_stop,
        )

        # Prewarm and measure TTFA
        self._prewarm_engine()

        # Callback to be set externally if needed
        self.on_first_audio_chunk_synthesize: Optional[Callable[[], None]] = None

    def _prewarm_engine(self):
        """Prewarms the engine and measures Time To First Audio (TTFA)."""
        # Prewarm
        self.stream.feed("prewarm")
        play_kwargs = dict(
            log_synthesized_text=False,
            muted=True,
            fast_sentence_fragment=False,
            comma_silence_duration=SILENCE_DURATIONS["comma"],
            sentence_silence_duration=SILENCE_DURATIONS["sentence"],
            default_silence_duration=SILENCE_DURATIONS["default"],
            force_first_fragment_after_words=999999,
        )
        self.stream.play(**play_kwargs)
        while self.stream.is_playing():
            time.sleep(0.01)
        self.finished_event.wait()
        self.finished_event.clear()

        # Measure TTFA
        start_time = time.time()
        ttfa = None

        def on_audio_chunk_ttfa(_: bytes):
            nonlocal ttfa
            if ttfa is None:
                ttfa = time.time() - start_time
                logger.debug(f"TTFA measurement first chunk arrived, TTFA: {ttfa:.2f}s.")

        self.stream.feed("This is a test sentence to measure the time to first audio chunk.")
        play_kwargs_ttfa = dict(
            on_audio_chunk=on_audio_chunk_ttfa,
            log_synthesized_text=False,
            muted=True,
            fast_sentence_fragment=False,
            comma_silence_duration=SILENCE_DURATIONS["comma"],
            sentence_silence_duration=SILENCE_DURATIONS["sentence"],
            default_silence_duration=SILENCE_DURATIONS["default"],
            force_first_fragment_after_words=999999,
        )
        self.stream.play_async(**play_kwargs_ttfa)

        while ttfa is None and (self.stream.is_playing() or not self.finished_event.is_set()):
            time.sleep(0.01)
        self.stream.stop()

        if not self.finished_event.is_set():
            self.finished_event.wait(timeout=2.0)
        self.finished_event.clear()

        if ttfa is not None:
            logger.debug(f"TTFA measurement complete. TTFA: {ttfa:.2f}s.")
            self.tts_inference_time = ttfa * 1000  # Store as ms
        else:
            logger.warning("TTFA measurement failed (no audio chunk received).")
            self.tts_inference_time = 0

    def on_audio_stream_stop(self) -> None:
        """
        Callback executed when the RealtimeTTS audio stream stops processing.
        """
        logger.debug("Audio stream stopped.")
        self.finished_event.set()

    def synthesize_generator(
        self,
        generator: Generator[str, None, None],
        stop_event: Optional[threading.Event] = None,
        generation_string: str = "",
    ) -> bool:
        """
        Synthesizes audio from a generator yielding text chunks.

        EdgeTTS plays audio directly via MPV (no queue).

        Args:
            generator: A generator yielding text chunks (strings) to synthesize.
            stop_event: A threading.Event to signal interruption of the synthesis.
                        If None, uses self.stop_event.
            generation_string: An optional identifier string for logging purposes.

        Returns:
            True if synthesis completed fully, False if interrupted by stop_event.
        """
        if stop_event is None:
            stop_event = self.stop_event

        # Feed the generator to the stream
        self.stream.feed(generator)
        self.finished_event.clear()

        # Callback state
        callback_fired = False

        def on_playback_start():
            """Called when audio playback starts."""
            nonlocal callback_fired
            if not callback_fired and self.on_first_audio_chunk_synthesize:
                try:
                    logger.debug(f"{generation_string} Firing on_first_audio_chunk_synthesize.")
                    self.on_first_audio_chunk_synthesize()
                except Exception as e:
                    logger.error(
                        f"{generation_string} Error in on_first_audio_chunk_synthesize callback: {e}",
                        exc_info=True,
                    )
                callback_fired = True

        play_kwargs = dict(
            log_synthesized_text=True,
            muted=False,  # Direct playback via MPV
            fast_sentence_fragment=False,
            comma_silence_duration=SILENCE_DURATIONS["comma"],
            sentence_silence_duration=SILENCE_DURATIONS["sentence"],
            default_silence_duration=SILENCE_DURATIONS["default"],
            force_first_fragment_after_words=999999,
        )

        logger.debug(f"▶️ {generation_string} Starting synthesis from generator.")

        # Set the callback
        self.engine.on_playback_started = on_playback_start

        self.stream.play_async(**play_kwargs)

        # Wait loop for completion or interruption
        while self.stream.is_playing() or not self.finished_event.is_set():
            if stop_event.is_set():
                self.stream.stop()
                logger.debug(f"{generation_string} Synthesis aborted by stop_event.")
                self.finished_event.wait(timeout=1.0)
                return False
            time.sleep(0.01)

        logger.debug(f"{generation_string} Synthesis complete.")
        return True
