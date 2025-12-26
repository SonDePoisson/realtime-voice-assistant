# main_bis.py - Version Terminal (sans web/WebSocket)
import logging
import signal
import threading
import time
import asyncio

from logsetup import setup_logging

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

logger.info("🖥️👋 Démarrage de l'assistant vocal en mode terminal")

import pyaudio
from colors import Colors
from speech_pipeline_manager import SpeechPipelineManager
from audio_in import AudioInputProcessor

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
TTS_START_ENGINE = "edgeTTS"
LLM_START_PROVIDER = "ollama"
LLM_START_MODEL = "ministral-3"
LANGUAGE = "fr"

# Configuration PyAudio
SAMPLE_RATE = 48000  # AudioInputProcessor attend 48kHz (resample vers 16kHz)
CHUNK_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16


# --------------------------------------------------------------------
# Callbacks pour le terminal
# --------------------------------------------------------------------
class TerminalCallbacks:
    """Gère les callbacks de transcription pour le mode terminal."""

    def __init__(self, pipeline: SpeechPipelineManager, audio_processor: AudioInputProcessor):
        self.pipeline = pipeline
        self.audio_processor = audio_processor
        self.final_transcription = ""
        self.partial_transcription = ""
        self.user_finished_turn = False
        self.tts_playing = False
        self.interruption_time = 0.0

    def on_partial(self, txt: str):
        """Callback pour les transcriptions partielles."""
        self.partial_transcription = txt
        print(f"\r{Colors.CYAN}[Vous]: {txt}{Colors.RESET}".ljust(80), end="", flush=True)

    def on_potential_sentence(self, txt: str):
        """Callback quand une phrase potentielle est détectée."""
        logger.debug(f"🎙️ Phrase potentielle: '{txt}'")
        self.pipeline.prepare_generation(txt)

    def on_potential_final(self, txt: str):
        """Callback quand on approche de la fin de la transcription."""
        logger.info(f"{Colors.MAGENTA}🎙️ HOT: {txt}{Colors.RESET}")

    def on_before_final(self, audio: bytes, txt: str):
        """Callback juste avant la transcription finale."""
        print()  # Nouvelle ligne après le partial
        logger.info(f"{Colors.GREEN}🎙️ Fin du tour utilisateur{Colors.RESET}")
        self.user_finished_turn = True

        # Bloquer le micro pendant le TTS
        if not self.audio_processor.interrupted:
            logger.info(f"{Colors.CYAN}🎙️ ⏸️ Microphone interrompu (fin de tour){Colors.RESET}")
            self.audio_processor.interrupted = True
            self.interruption_time = time.time()

        # Permettre la synthèse TTS
        if self.pipeline.is_valid_gen():
            self.pipeline.running_generation.tts_quick_allowed_event.set()

        # Ajouter à l'historique
        user_text = self.final_transcription if self.final_transcription else self.partial_transcription
        if user_text:
            logger.info(f"🎙️ Ajout à l'historique: '{user_text}'")
            self.pipeline.history.append({"role": "user", "content": user_text})

    def on_final(self, txt: str):
        """Callback pour la transcription finale."""
        self.final_transcription = txt
        print(f"{Colors.GREEN}[Vous]: {txt}{Colors.RESET}")
        self.partial_transcription = ""

    def on_recording_start(self):
        """Callback quand l'enregistrement commence."""
        logger.info(f"{Colors.ORANGE}🎙️ Enregistrement démarré{Colors.RESET}")

        # Si le TTS jouait, on interrompt
        if self.tts_playing:
            logger.info(f"{Colors.RED}🛑 Interruption du TTS par l'utilisateur{Colors.RESET}")
            self.pipeline.abort_generation(reason="User interrupted")
            self.tts_playing = False

    def on_silence_active(self, is_silent: bool):
        """Callback quand l'état de silence change."""
        pass

    def on_partial_assistant_text(self, txt: str):
        """Callback pour le texte partiel de l'assistant."""
        print(f"\r{Colors.YELLOW}[Assistant]: {txt}{Colors.RESET}".ljust(80), end="", flush=True)

    def send_final_assistant_answer(self):
        """Envoie la réponse finale de l'assistant."""
        if self.pipeline.is_valid_gen():
            answer = self.pipeline.running_generation.quick_answer + self.pipeline.running_generation.final_answer
            if answer:
                print(f"\n{Colors.GREEN}[Assistant]: {answer}{Colors.RESET}")
                self.pipeline.history.append({"role": "assistant", "content": answer})


# --------------------------------------------------------------------
# Capture microphone avec PyAudio
# --------------------------------------------------------------------
class MicrophoneCapture:
    """Capture le microphone et met les chunks dans une queue."""

    def __init__(self, audio_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.audio_queue = audio_queue
        self.loop = loop
        self.running = False
        self.thread = None
        self.pyaudio = None
        self.stream = None

    def start(self):
        """Démarre la capture du microphone."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("🎤 Capture microphone démarrée")

    def stop(self):
        """Arrête la capture du microphone."""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pyaudio:
            self.pyaudio.terminate()
        logger.info("🎤 Capture microphone arrêtée")

    def _capture_loop(self):
        """Boucle de capture dans un thread séparé."""
        try:
            self.pyaudio = pyaudio.PyAudio()
            self.stream = self.pyaudio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )

            while self.running:
                try:
                    data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    # Créer le dictionnaire de métadonnées comme attendu par AudioInputProcessor
                    audio_data = {"pcm": data}
                    # Mettre dans la queue de façon thread-safe
                    self.loop.call_soon_threadsafe(lambda d=audio_data: self.audio_queue.put_nowait(d))
                except Exception as e:
                    if self.running:
                        logger.error(f"🎤 Erreur capture: {e}")
                    break

        except Exception as e:
            logger.error(f"🎤 Erreur initialisation PyAudio: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.pyaudio:
                self.pyaudio.terminate()


# --------------------------------------------------------------------
# Fonction principale async
# --------------------------------------------------------------------
async def main_async(pipeline: SpeechPipelineManager):
    """Point d'entrée principal asynchrone."""
    # Initialiser AudioInputProcessor ICI (nécessite event loop actif)
    audio_processor = AudioInputProcessor(language=LANGUAGE)

    # Créer les callbacks
    callbacks = TerminalCallbacks(pipeline, audio_processor)

    # Configurer les callbacks sur l'AudioInputProcessor
    audio_processor.realtime_callback = callbacks.on_partial
    audio_processor.transcriber.potential_sentence_end = callbacks.on_potential_sentence
    audio_processor.transcriber.potential_full_transcription_callback = callbacks.on_potential_final
    audio_processor.transcriber.full_transcription_callback = callbacks.on_final
    audio_processor.transcriber.before_final_sentence = callbacks.on_before_final
    audio_processor.recording_start_callback = callbacks.on_recording_start
    audio_processor.silence_active_callback = callbacks.on_silence_active

    # Callback pour le texte partiel de l'assistant
    pipeline.on_partial_assistant_text = callbacks.on_partial_assistant_text

    # Queue pour les chunks audio
    audio_queue = asyncio.Queue()

    # Capture microphone
    loop = asyncio.get_event_loop()
    mic_capture = MicrophoneCapture(audio_queue, loop)

    # Gestion de l'arrêt propre
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info("\n🛑 Arrêt demandé (Ctrl+C)")
        loop.call_soon_threadsafe(shutdown_event.set)

    signal.signal(signal.SIGINT, signal_handler)

    logger.info(f"{Colors.GREEN}✅ Prêt ! Parlez dans le microphone. Ctrl+C pour quitter.{Colors.RESET}")
    print("-" * 60)

    # Démarrer la capture micro
    mic_capture.start()

    # Lancer le traitement des chunks audio
    audio_task = asyncio.create_task(audio_processor.process_chunk_queue(audio_queue))

    try:
        while not shutdown_event.is_set():
            await asyncio.sleep(0.1)

            # Reset du flag interrupted après un délai
            if (
                audio_processor.interrupted
                and callbacks.interruption_time
                and time.time() - callbacks.interruption_time > 2.0
            ):
                logger.info(f"{Colors.CYAN}🎙️ ▶️ Microphone réactivé{Colors.RESET}")
                audio_processor.interrupted = False
                callbacks.interruption_time = 0

            # Vérifier si le TTS a commencé
            if (
                pipeline.running_generation
                and pipeline.running_generation.quick_answer_first_chunk_ready
                and not callbacks.tts_playing
            ):
                callbacks.tts_playing = True
                logger.info(f"{Colors.BLUE}🔊 TTS démarré{Colors.RESET}")

            # Vérifier si une génération est terminée
            if (
                pipeline.running_generation
                and pipeline.running_generation.audio_quick_finished
                and not pipeline.running_generation.abortion_started
            ):
                if (
                    pipeline.running_generation.audio_final_finished
                    or not pipeline.running_generation.quick_answer_provided
                ):
                    callbacks.send_final_assistant_answer()
                    pipeline.running_generation = None
                    callbacks.tts_playing = False

                    # Réactiver le micro après la fin du TTS
                    if audio_processor.interrupted:
                        logger.info(f"{Colors.CYAN}🎙️ ▶️ Microphone réactivé (fin TTS){Colors.RESET}")
                        audio_processor.interrupted = False
                        callbacks.interruption_time = 0

    except asyncio.CancelledError:
        pass
    finally:
        logger.info("🧹 Nettoyage...")
        mic_capture.stop()
        audio_task.cancel()
        try:
            await audio_task
        except asyncio.CancelledError:
            pass
        audio_processor.shutdown()


def main():
    """Point d'entrée principal."""
    logger.info("🚀 Initialisation des composants...")

    # Initialiser le pipeline de synthèse vocale AVANT la boucle async
    # (EdgeEngine utilise asyncio.run() en interne)
    pipeline = SpeechPipelineManager(
        tts_engine=TTS_START_ENGINE,
        llm_provider=LLM_START_PROVIDER,
        llm_model=LLM_START_MODEL,
    )

    try:
        asyncio.run(main_async(pipeline))
    finally:
        # Cleanup
        pipeline.shutdown()
        logger.info("👋 Au revoir !")


if __name__ == "__main__":
    main()
