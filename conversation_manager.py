import logging
import os
import sys

import threading
import queue
import time
from typing import Optional

from stt_module import TranscriptionProcessor, USE_TURN_DETECTION
from tts_module import AudioProcessor
from llm_module import LLM

# Configure logging
# Use the root logger configured by the main application if available, else basic config
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
# Check if root logger already has handlers (likely configured by main app)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )  # Default to stdout if not configured
logger = logging.getLogger(__name__)  # Get logger for this module
logger.setLevel(log_level)  # Ensure module logger respects level


class Generation:
    """État d'une génération unique (une réponse de l'assistant)"""

    def __init__(self, user_text: str):
        self.user_text = user_text
        self.assistant_text = ""
        self.text_queue = queue.Queue()  # Queue pour streaming LLM → TTS
        self.audio_queue = queue.Queue()  # Queue pour streaming TTS → Player
        self.llm_completed = False
        self.tts_started = False
        self.tts_completed = False
        self.audio_started = False
        self.audio_completed = False
        self.timestamp = time.time()


class ConversationManager:
    """
    Orchestre la conversation en temps réel avec 3 composants principaux:
    - STT (Speech-to-Text) avec Whisper small
    - LLM (Large Language Model) avec Ollama llama3.2:3b
    - TTS (Text-to-Speech) avec Kokoro voix française

    Architecture simplifiée à 2 worker threads:
    - Thread LLM: génère les réponses en streaming
    - Thread TTS: synthétise et joue l'audio
    """

    def __init__(
        self,
        llm_provider: str = "ollama",
        llm_model: str = "llama3.2:3b",
        tts_engine: str = "kokoro",
        system_prompt_file: str = "system_prompt.txt",
    ):
        """
        Initialise le gestionnaire de conversation.

        Args:
            llm_provider: provider LLM ("ollama")
            llm_model: Nom du modèle LLM
            tts_engine: Moteur TTS ("kokoro", "orpheus")
            system_prompt_file: Chemin vers le fichier de prompt système
        """
        logger.debug("Initialisation du ConversationManager...")

        # Charger le prompt système
        self.system_prompt = self._load_system_prompt(system_prompt_file)

        # Initialiser les composants
        logger.debug("Initialisation du LLM...")
        self.llm = LLM(
            provider=llm_provider,
            model=llm_model,
            system_prompt=self.system_prompt,
            no_think=False,
        )

        logger.debug("Initialisation du TTS...")
        self.tts = AudioProcessor(engine=tts_engine)

        logger.debug("Initialisation du STT...")
        self.stt = TranscriptionProcessor(source_language="fr")

        # Connecter les callbacks STT
        self.stt.full_transcription_callback = self._on_user_input
        self.stt.on_recording_start = self._on_user_interrupt

        # Turn detection (si activé)
        if USE_TURN_DETECTION:
            logger.debug("Turn detection activé")
            # Le turn detection est déjà géré dans TranscriptionProcessor

        # État de la conversation
        self.current_generation: Optional[Generation] = None
        self.history = []  # Historique des messages pour le LLM

        # Events de synchronisation
        self.abort_event = threading.Event()
        self.new_input_event = threading.Event()
        self.shutdown_event = threading.Event()

        # Worker threads
        self.llm_thread = None
        self.tts_thread = None
        self.audio_player_thread = None

        logger.debug("ConversationManager initialisé")

    def _load_system_prompt(self, filepath: str) -> str:
        """Charge le prompt système depuis un fichier"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
                logger.debug(f"Prompt système chargé depuis {filepath}")
                return prompt
        except FileNotFoundError:
            logger.warning(
                f"Fichier {filepath} introuvable, utilisation du prompt par défaut"
            )
            return "Tu es un assistant vocal français serviable et amical."

    def _start_workers(self):
        """Démarre les threads workers"""
        logger.debug("Démarrage des workers...")

        self.llm_thread = threading.Thread(
            target=self._llm_worker, name="LLM-Worker", daemon=True
        )
        self.llm_thread.start()

        self.tts_thread = threading.Thread(
            target=self._tts_worker, name="TTS-Worker", daemon=True
        )
        self.tts_thread.start()

        self.audio_player_thread = threading.Thread(
            target=self._audio_player_worker, name="Audio-Player", daemon=True
        )
        self.audio_player_thread.start()

        logger.debug("Workers démarrés (LLM, TTS, Audio Player)")

    def _on_user_input(self, text: str):
        """
        Callback appelé quand le STT a finalisé une transcription.

        Args:
            text: Texte transcrit de l'utilisateur
        """
        if not text or not text.strip():
            return

        text = text.strip()
        logger.debug(f"Utilisateur: {text}")

        # Interrompre la génération en cours si elle existe
        self._abort_current()

        # Créer une nouvelle génération
        self.current_generation = Generation(text)

        # Ajouter à l'historique
        self.history.append({"role": "user", "content": text})

        # Signaler le worker LLM
        self.new_input_event.set()

    def _on_user_interrupt(self):
        """Callback appelé quand l'utilisateur commence à parler (interruption)"""
        logger.debug("Interruption détectée")
        self._abort_current()

    def _llm_worker(self):
        """
        Thread worker: génère les réponses du LLM en streaming.

        Lit les nouvelles entrées utilisateur et génère les réponses LLM,
        en plaçant chaque chunk de texte dans la queue pour le TTS.
        """
        logger.debug("LLM Worker démarré")

        while not self.shutdown_event.is_set():
            # Attendre un nouvel input avec timeout
            triggered = self.new_input_event.wait(timeout=1.0)

            if not triggered:
                continue

            self.new_input_event.clear()

            gen = self.current_generation
            if not gen:
                continue

            logger.debug("Génération LLM en cours...")

            try:
                # Générer la réponse en streaming
                for chunk in self.llm.generate(
                    text=gen.user_text,
                    history=self.history[:-1],  # Historique sans le message actuel
                    use_system_prompt=True,
                ):
                    # Vérifier si on doit arrêter
                    if self.abort_event.is_set():
                        logger.debug("Génération LLM annulée")
                        break

                    # Ajouter le chunk à la queue pour TTS
                    gen.text_queue.put(chunk)
                    gen.assistant_text += chunk

                # Marquer la génération LLM comme terminée
                if not self.abort_event.is_set():
                    gen.llm_completed = True
                    gen.text_queue.put(None)  # Signal de fin

                    # Ajouter la réponse complète à l'historique
                    self.history.append(
                        {"role": "assistant", "content": gen.assistant_text}
                    )

                    logger.debug(f"Assistant: {gen.assistant_text}")

                    # Limiter la taille de l'historique
                    if len(self.history) > 20:
                        self.history = self.history[-20:]

            except Exception as e:
                logger.error(f"Erreur LLM: {e}", exc_info=True)
                gen.text_queue.put(None)  # Signal de fin en cas d'erreur

    def _tts_worker(self):
        """
        Thread worker: synthétise l'audio à partir des chunks de texte.

        Attend que des chunks de texte soient disponibles dans text_queue,
        puis les synthétise en audio et les place dans audio_queue.
        """
        logger.debug("TTS Worker démarré")

        while not self.shutdown_event.is_set():
            time.sleep(0.01)  # Petite pause pour éviter de saturer le CPU

            gen = self.current_generation
            if not gen or gen.tts_started:
                continue

            # Attendre qu'au moins un chunk soit disponible
            if gen.text_queue.empty():
                continue

            gen.tts_started = True
            logger.debug("Synthèse TTS démarrée...")

            # Générateur qui consomme la queue de texte
            def text_chunks():
                while True:
                    try:
                        chunk = gen.text_queue.get(timeout=0.1)
                        if chunk is None:
                            break
                        if self.abort_event.is_set():
                            break
                        yield chunk
                    except queue.Empty:
                        # Vérifier si le LLM a terminé
                        if gen.llm_completed:
                            break
                        continue

            try:
                # Synthétiser dans la queue audio (pas de lecture directe)
                completed = self.tts.synthesize_generator(
                    text_chunks(),
                    audio_chunks=gen.audio_queue,  # Mettre l'audio dans la queue
                    stop_event=self.abort_event,
                )

                if completed and not self.abort_event.is_set():
                    gen.tts_completed = True
                    gen.audio_queue.put(None)  # Signal de fin pour l'audio player
                    logger.debug("Synthèse TTS terminée")
                else:
                    gen.audio_queue.put(None)  # Signal de fin même si interrompu
                    logger.debug("Synthèse TTS interrompue")

            except Exception as e:
                logger.error(f"Erreur TTS: {e}", exc_info=True)
                gen.audio_queue.put(None)  # Signal de fin en cas d'erreur

    def _audio_player_worker(self):
        """
        Thread worker: joue l'audio depuis la queue audio.

        Attend que des chunks audio soient disponibles dans audio_queue,
        puis les joue sur les haut-parleurs via pyaudio.
        """
        import pyaudio

        logger.debug("Audio Player Worker démarré")

        # Initialiser PyAudio une seule fois (réutilisé pour toutes les générations)
        p = pyaudio.PyAudio()

        try:
            while not self.shutdown_event.is_set():
                time.sleep(0.01)

                gen = self.current_generation
                if not gen or gen.audio_started:
                    continue

                # Attendre qu'au moins quelques chunks audio soient disponibles (buffer)
                if gen.audio_queue.qsize() < 5 and not gen.tts_completed:
                    continue

                gen.audio_started = True
                logger.debug("Lecture audio démarrée...")

                # Ouvrir un nouveau stream pour cette génération
                stream = None
                try:
                    stream = p.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=24000,
                        output=True,
                        frames_per_buffer=4096,
                    )

                    # Lire et jouer les chunks audio
                    while not self.abort_event.is_set():
                        try:
                            chunk = gen.audio_queue.get(timeout=0.1)
                            if chunk is None:  # Signal de fin
                                break
                            stream.write(chunk)
                        except queue.Empty:
                            if gen.tts_completed:
                                break
                            continue

                    if not self.abort_event.is_set():
                        gen.audio_completed = True
                        logger.debug("Lecture audio terminée")
                    else:
                        logger.debug("Lecture audio interrompue")

                except Exception as e:
                    logger.error(f"Erreur lors de la lecture audio: {e}", exc_info=True)

                finally:
                    # Fermer le stream proprement après chaque génération
                    if stream:
                        stream.stop_stream()
                        stream.close()
                        logger.debug("Stream audio fermé")

        finally:
            # Cleanup PyAudio
            p.terminate()
            logger.debug("Audio Player Worker arrêté")

    def _abort_current(self):
        """Interrompt la génération en cours"""
        if not self.current_generation:
            return

        logger.debug("Interruption de la génération en cours...")

        # Signaler l'interruption
        self.abort_event.set()

        # Annuler la génération LLM
        try:
            self.llm.cancel_generation()
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation LLM: {e}")

        # Arrêter le TTS
        try:
            if hasattr(self.tts, "stream"):
                self.tts.stream.stop()
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt TTS: {e}")

        # Vider les queues de texte et audio
        while not self.current_generation.text_queue.empty():
            try:
                self.current_generation.text_queue.get_nowait()
            except queue.Empty:
                break

        while not self.current_generation.audio_queue.empty():
            try:
                self.current_generation.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Petite pause pour laisser les workers réagir
        time.sleep(0.1)

        # Clear l'event d'interruption
        self.abort_event.clear()

        logger.debug("Interruption complète")

    def start(self):
        """Démarre le système de conversation"""
        logger.debug("Démarrage de l'assistant vocal...")

        # Démarrer les workers
        self._start_workers()

        # Démarrer le STT (écoute du microphone)
        logger.debug("Démarrage de l'écoute...")
        self.stt.transcribe_loop()

        logger.debug("Assistant vocal prêt!")

    def shutdown(self):
        """Arrête proprement le système"""
        logger.debug("Arrêt de l'assistant...")

        # Signaler l'arrêt
        self.shutdown_event.set()

        # Arrêter les composants
        try:
            if self.stt:
                self.stt.shutdown()
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt STT: {e}")

        try:
            if self.tts and hasattr(self.tts, "stream"):
                self.tts.stream.stop()
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt TTS: {e}")

        # Attendre les threads
        if self.llm_thread:
            self.llm_thread.join(timeout=2.0)
        if self.tts_thread:
            self.tts_thread.join(timeout=2.0)
        if self.audio_player_thread:
            self.audio_player_thread.join(timeout=2.0)

        logger.debug("Assistant arrêté")
