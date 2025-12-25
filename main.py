#!/usr/bin/env python3
"""
Voice-enabled conversational agent using SpeechPipelineManager and TranscriptionProcessor.

This script provides both voice and text interfaces for interacting with an LLM.
"""

import logging
import signal
import sys
import time
import threading

from logsetup import setup_logging
from speech_pipeline_manager import SpeechPipelineManager
from transcribe import TranscriptionProcessor
from colors import Colors

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceAgent:
    """
    A voice-enabled conversational agent that uses:
    - TranscriptionProcessor for STT (Speech-to-Text)
    - SpeechPipelineManager for LLM inference and TTS synthesis
    """

    def __init__(
        self,
        tts_engine: str = "edgeTTS",
        llm_provider: str = "ollama",
        llm_model: str = "ministral-3",
        language: str = "fr",
    ):
        """
        Initialize the voice agent.

        Args:
            tts_engine: TTS engine to use (default: "edgeTTS")
            llm_provider: LLM provider (default: "ollama")
            llm_model: LLM model name (default: "ministral-3")
            language: Language for STT (default: "fr")
        """
        self.running = True
        self.is_listening = False
        self.is_speaking = False
        self.current_partial_text = ""
        self.language = language

        logger.info(f"Initializing VoiceAgent with {llm_provider}/{llm_model}...")

        # Initialize the speech pipeline manager (LLM + TTS)
        self.pipeline = SpeechPipelineManager(
            tts_engine=tts_engine,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

        # Get pipeline latency for STT configuration
        pipeline_latency_ms = self.pipeline.full_output_pipeline_latency
        pipeline_latency_s = pipeline_latency_ms / 1000.0

        # Initialize STT with microphone enabled
        self.stt = TranscriptionProcessor(
            source_language=language,
            realtime_transcription_callback=self._on_realtime_transcription,
            full_transcription_callback=self._on_full_transcription,
            on_recording_start_callback=self._on_recording_start,
            silence_active_callback=self._on_silence_active,
            pipeline_latency=pipeline_latency_s,
            recorder_config={
                "use_microphone": True,  # Enable microphone input
                "spinner": False,
                "model": "small",
                "realtime_model_type": "small",
                "language": language,
                "silero_sensitivity": 0.05,
                "webrtc_sensitivity": 3,
                "post_speech_silence_duration": 0.7,
                "min_length_of_recording": 0.5,
                "min_gap_between_recordings": 0,
                "enable_realtime_transcription": True,
                "realtime_processing_pause": 0.03,
                "silero_use_onnx": True,
                "silero_deactivity_detection": True,
                "beam_size": 3,
                "beam_size_realtime": 3,
                "no_log_file": True,
                "debug_mode": False,
            },
        )

        # Start the transcription loop
        self.transcription_thread = threading.Thread(
            target=self._run_transcription_loop,
            name="TranscriptionLoopThread",
            daemon=True,
        )
        self.transcription_thread.start()

        logger.info("VoiceAgent initialized successfully.")

    def _on_realtime_transcription(self, text: str) -> None:
        """Callback for real-time (partial) transcription updates."""
        if text != self.current_partial_text:
            self.current_partial_text = text
            # Clear line and show partial transcription
            print(f"\r{Colors.YELLOW}You: {Colors.CYAN}{text}{Colors.RESET}    ", end="", flush=True)

    def _on_full_transcription(self, text: str) -> None:
        """Callback for final transcription - triggers LLM response."""
        if not text or not text.strip():
            return

        self.current_partial_text = ""
        self.is_listening = False

        # Show final user text
        print(f"\r{Colors.YELLOW}You: {Colors.RESET}{text}                    ")
        logger.info(f"Full transcription received: {text}")

        # Send to LLM pipeline
        self._process_user_input(text)

    def _on_recording_start(self) -> None:
        """Callback when recording starts (voice activity detected)."""
        self.is_listening = True
        # If assistant is speaking, interrupt
        if self.is_speaking:
            logger.info("User started speaking, interrupting assistant...")
            self.pipeline.abort_generation()
            self.is_speaking = False

    def _on_silence_active(self, is_active: bool) -> None:
        """Callback when silence detection state changes."""
        if is_active:
            logger.debug("Silence detected")
        else:
            logger.debug("Voice activity detected")

    def _run_transcription_loop(self) -> None:
        """Run the STT transcription loop in a background thread."""
        logger.info("Starting transcription loop...")
        try:
            self.stt.transcribe_loop()
        except Exception as e:
            logger.error(f"Transcription loop error: {e}", exc_info=True)

    def _process_user_input(self, user_input: str) -> None:
        """Process user input and generate response."""
        self.is_speaking = True

        # Prepare generation (sends to LLM + TTS pipeline)
        self.pipeline.prepare_generation(user_input)

        # Update history
        self.pipeline.history.append({"role": "user", "content": user_input})

        # Wait for response to complete
        response = self._wait_for_generation()

        if response:
            self.pipeline.history.append({"role": "assistant", "content": response})
            print(f"{Colors.GREEN}Assistant: {Colors.RESET}{response}\n")

        self.is_speaking = False

    def _wait_for_generation(self, timeout: float = 60.0) -> str:
        """Wait for the current generation to complete."""
        start_time = time.time()
        last_text = ""
        stable_count = 0

        while time.time() - start_time < timeout:
            # Check if user started speaking (interrupt)
            if self.is_listening:
                logger.info("User interrupted, aborting generation...")
                self.pipeline.abort_generation()
                return ""

            gen = self.pipeline.running_generation
            if gen is None:
                time.sleep(0.1)
                continue

            if gen.completed or gen.abortion_started:
                break

            if gen.llm_finished and gen.audio_quick_finished:
                if not gen.quick_answer_provided or gen.audio_final_finished:
                    break

            current_text = gen.quick_answer + gen.final_answer
            if current_text == last_text:
                stable_count += 1
                if stable_count > 20 and gen.llm_finished:
                    break
            else:
                stable_count = 0
                last_text = current_text

            time.sleep(0.1)

        gen = self.pipeline.running_generation
        if gen:
            return gen.quick_answer + gen.final_answer
        return ""

    def run(self) -> None:
        """Run the voice agent main loop."""
        print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}  Realtime Voice Assistant{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.GRAY}Speak into your microphone - I'm listening!")
        print(f"Press Ctrl+C to exit.{Colors.RESET}\n")

        print(f"{Colors.MAGENTA}[Listening...]{Colors.RESET}\n")

        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print(f"\n{Colors.GRAY}Shutting down...{Colors.RESET}")

    def shutdown(self) -> None:
        """Shutdown the agent and cleanup resources."""
        self.running = False
        logger.info("Shutting down VoiceAgent...")

        if hasattr(self, "stt") and self.stt:
            self.stt.shutdown()

        if hasattr(self, "pipeline") and self.pipeline:
            self.pipeline.shutdown()

        logger.info("VoiceAgent shutdown complete.")


class TextAgent:
    """
    A text-based conversational agent (fallback when no microphone).
    """

    def __init__(
        self,
        tts_engine: str = "edgeTTS",
        llm_provider: str = "ollama",
        llm_model: str = "ministral-3",
    ):
        self.running = True
        self.pipeline = SpeechPipelineManager(
            tts_engine=tts_engine,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        logger.info("TextAgent initialized.")

    def chat(self, user_input: str) -> str:
        """Send a message and get response."""
        self.pipeline.prepare_generation(user_input)
        response = self._wait_for_generation()
        if response:
            self.pipeline.history.append({"role": "user", "content": user_input})
            self.pipeline.history.append({"role": "assistant", "content": response})
        return response

    def _wait_for_generation(self, timeout: float = 60.0) -> str:
        """Wait for generation to complete."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            gen = self.pipeline.running_generation
            if gen is None:
                time.sleep(0.1)
                continue
            if gen.completed or gen.abortion_started:
                break
            if gen.llm_finished and gen.audio_quick_finished:
                if not gen.quick_answer_provided or gen.audio_final_finished:
                    break
            time.sleep(0.1)

        gen = self.pipeline.running_generation
        return (gen.quick_answer + gen.final_answer) if gen else ""

    def run(self) -> None:
        """Run the text-based REPL loop."""
        print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}  Realtime Voice Assistant - Text Mode{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.GRAY}Type your message and press Enter.")
        print(f"Commands: /quit, /reset{Colors.RESET}\n")

        while self.running:
            try:
                user_input = input(f"{Colors.YELLOW}You: {Colors.RESET}").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["/quit", "/exit", "/q"]:
                    print(f"{Colors.GRAY}Goodbye!{Colors.RESET}")
                    break
                elif user_input.lower() == "/reset":
                    self.pipeline.reset()
                    print(f"{Colors.GRAY}Conversation reset.{Colors.RESET}")
                    continue

                response = self.chat(user_input)
                if response:
                    print(f"{Colors.GREEN}Assistant: {Colors.RESET}{response}\n")
                else:
                    print(f"{Colors.RED}[No response]{Colors.RESET}\n")

            except KeyboardInterrupt:
                print(f"\n{Colors.GRAY}Type /quit to exit.{Colors.RESET}")
            except EOFError:
                break

    def shutdown(self) -> None:
        """Shutdown the agent."""
        self.running = False
        self.pipeline.shutdown()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Voice-enabled conversational agent")
    parser.add_argument("--model", "-m", default="ministral-3", help="LLM model (default: ministral-3)")
    parser.add_argument("--provider", "-p", default="ollama", help="LLM provider (default: ollama)")
    parser.add_argument("--tts", default="edgeTTS", help="TTS engine (default: edgeTTS)")
    parser.add_argument("--language", "-l", default="fr", help="STT language (default: fr)")
    parser.add_argument("--text", "-t", action="store_true", help="Use text mode instead of voice")

    args = parser.parse_args()

    agent = None

    def signal_handler(sig, frame):
        print(f"\n{Colors.GRAY}Shutting down...{Colors.RESET}")
        if agent:
            agent.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if args.text:
            agent = TextAgent(
                tts_engine=args.tts,
                llm_provider=args.provider,
                llm_model=args.model,
            )
        else:
            agent = VoiceAgent(
                tts_engine=args.tts,
                llm_provider=args.provider,
                llm_model=args.model,
                language=args.language,
            )
        agent.run()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        if agent:
            agent.shutdown()


if __name__ == "__main__":
    main()
