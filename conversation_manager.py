import logging

import threading
import queue
import time
from typing import Optional

from stt_module import TranscriptionProcessor, USE_TURN_DETECTION
from tts_module import AudioProcessor
from llm_module import LLM

logger = logging.getLogger(__name__)


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
        # llm_model: str = "ministral-3:latest",
        tts_engine: str = "kokoro",
        language: str = "fr",
        system_prompt_file: str = "system_prompt.txt",
    ):
        """
        Initialise le gestionnaire de conversation.

        Args:
            llm_provider: provider LLM ("ollama")
            llm_model: Nom du modèle LLM
            tts_engine: Moteur TTS ("kokoro", "edge_tts")
            system_prompt_file: Chemin vers le fichier de prompt système
        """
        logger.debug("-- [Init] ConversationManager")

        # Charger le prompt système
        self.system_prompt = self._load_system_prompt(system_prompt_file)

        # Initialiser les composants
        logger.debug("-- [Init] LLM")
        self.llm = LLM(
            provider=llm_provider,
            model=llm_model,
            system_prompt=self.system_prompt,
            no_think=False,
        )

        logger.debug("-- [Init] TTS")
        self.tts = AudioProcessor(engine=tts_engine, language=language)

        logger.debug("-- [Init] STT")
        self.stt = TranscriptionProcessor(source_language=language)

        # Connecter les callbacks STT
        self.stt.full_transcription_callback = self._on_user_input
        self.stt.on_recording_start_callback = self._on_user_interrupt
        self.stt.silence_active_callback = self._on_silence_state_changed

        # TODO : VERIFY TURN DETECTION
        # Turn detection #
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

        logger.debug("-- [Init] -- DONE")

    def _load_system_prompt(self, filepath: str) -> str:
        """Charge le prompt système depuis un fichier"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
                logger.debug(f"Prompt système chargé depuis {filepath}")
                return prompt
        except FileNotFoundError:
            logger.warning(f"Fichier {filepath} introuvable, utilisation du prompt par défaut")
            return "Tu es un assistant vocal serviable et amical."

    def _start_workers(self):
        """Démarre les threads workers"""
        logger.debug("-- [Init] Threads")
        self.llm_thread = threading.Thread(target=self._llm_worker, name="LLM-Worker", daemon=True)
        self.llm_thread.start()

        self.tts_thread = threading.Thread(target=self._tts_worker, name="TTS-Worker", daemon=True)
        self.tts_thread.start()

        self.audio_player_thread = threading.Thread(target=self._audio_player_worker, name="Audio-Player", daemon=True)
        self.audio_player_thread.start()

    def _on_user_input(self, text: str):
        """
        Callback appelé quand le STT a finalisé une transcription.

        Args:
            text: Texte transcrit de l'utilisateur
        """
        if not text or not text.strip():
            return

        text = text.strip()
        logger.info(f"--> USER: {text}")

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
        logger.debug("-- [INTERRUPTION] Recording started")
        self._abort_current()

    def _on_silence_state_changed(self, silence_active: bool):
        """
        Callback appelé quand l'état du silence change.

        Args:
            silence_active: True si le silence est actif, False si l'utilisateur parle
        """
        if not silence_active:
            # L'utilisateur vient de commencer à parler
            # Interrompre la génération en cours si elle existe
            if self.current_generation and not self.current_generation.audio_completed:
                logger.debug("-- [INTERRUPTION] User started speaking, aborting current generation")
                self._abort_current()

    def _llm_worker(self):
        """
        Thread worker: génère les réponses du LLM en streaming.

        Lit les nouvelles entrées utilisateur et génère les réponses LLM,
        en plaçant chaque chunk de texte dans la queue pour le TTS.
        """
        logger.debug("-- [START] Workers")

        while not self.shutdown_event.is_set():
            # Attendre un nouvel input avec timeout
            triggered = self.new_input_event.wait(timeout=1.0)

            if not triggered:
                continue

            self.new_input_event.clear()

            gen = self.current_generation
            if not gen:
                continue

            try:
                # Générer la réponse en streaming
                for chunk in self.llm.generate(
                    text=gen.user_text,
                    history=self.history[:-1],  # Historique sans le message actuel
                    use_system_prompt=True,
                ):
                    # Vérifier si on doit arrêter
                    if self.abort_event.is_set():
                        logger.debug("-- [INTERRUPTION] LLM generation aborted")
                        break

                    # Ajouter le chunk à la queue pour TTS
                    gen.text_queue.put(chunk)
                    gen.assistant_text += chunk

                # Marquer la génération LLM comme terminée
                if not self.abort_event.is_set():
                    gen.llm_completed = True
                    gen.text_queue.put(None)  # Signal de fin

                    # Ajouter la réponse complète à l'historique
                    self.history.append({"role": "assistant", "content": gen.assistant_text})

                    logger.info(f"--> Assistant: {gen.assistant_text}")

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
        logger.debug("-- [START] TTS")

        while not self.shutdown_event.is_set():
            time.sleep(0.01)  # Petite pause pour éviter de saturer le CPU

            gen = self.current_generation
            if not gen or gen.tts_started:
                continue

            # Attendre qu'au moins un chunk soit disponible
            if gen.text_queue.empty():
                continue

            gen.tts_started = True

            # Générateur qui consomme la queue de texte
            def text_chunks():
                while True:
                    try:
                        chunk = gen.text_queue.get(timeout=0.1)
                        if chunk is None:
                            break
                        if self.abort_event.is_set():
                            logger.debug("-- [INTERRUPTION] TTS text streaming aborted")
                            break
                        yield chunk
                    except queue.Empty:
                        # Vérifier si le LLM a terminé
                        if gen.llm_completed:
                            break
                        continue

            try:
                # EdgeTTS joue directement via MPV (MP3/Opus), pas via queue+PyAudio
                # Les autres engines (Kokoro) utilisent la queue audio
                if self.tts.uses_direct_playback:
                    # Direct playback: pas de queue audio
                    completed = self.tts.synthesize_generator(
                        text_chunks(),
                        audio_chunks=None,  # Pas de queue, lecture directe
                        stop_event=self.abort_event,
                    )
                    # Pas besoin de signaler l'audio_player car il ne sera pas utilisé
                else:
                    # Queue audio: Synthétiser dans la queue pour PyAudio
                    completed = self.tts.synthesize_generator(
                        text_chunks(),
                        audio_chunks=gen.audio_queue,  # Mettre l'audio dans la queue
                        stop_event=self.abort_event,
                    )

                if completed and not self.abort_event.is_set():
                    gen.tts_completed = True
                    if not self.tts.uses_direct_playback:
                        gen.audio_queue.put(None)  # Signal de fin pour l'audio player
                    else:
                        # Direct playback: audio déjà joué, marquer comme terminé
                        gen.audio_started = True
                        gen.audio_completed = True
                    logger.debug("-- [DONE] TTS")
                else:
                    if not self.tts.uses_direct_playback:
                        gen.audio_queue.put(None)  # Signal de fin même si interrompu
                    else:
                        # Direct playback: audio interrompu
                        gen.audio_started = True
                        gen.audio_completed = False
                    logger.debug("-- [STOP] TTS")

            except Exception as e:
                logger.error(f"-- [ERROR] TTS: {e}", exc_info=True)
                if not self.tts.uses_direct_playback:
                    gen.audio_queue.put(None)  # Signal de fin en cas d'erreur

    def _audio_player_worker(self):
        """
        Thread worker: joue l'audio depuis la queue audio.

        Attend que des chunks audio soient disponibles dans audio_queue,
        puis les joue sur les haut-parleurs via pyaudio.

        Note: Si EdgeTTS est utilisé (direct playback), ce worker ne fait rien
        car l'audio est joué directement par RealtimeTTS via MPV.
        """
        import pyaudio

        logger.debug("-- [START] Audio")

        # Si EdgeTTS (direct playback), ce worker n'est pas nécessaire
        if self.tts.uses_direct_playback:
            logger.debug("-- [SKIP] Audio player (using direct playback via MPV)")
            while not self.shutdown_event.is_set():
                time.sleep(0.1)
            logger.debug("-- [SHUTDOWN] Audio Player (was idle)")
            return

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

                # Ouvrir un nouveau stream pour cette génération
                stream = None
                try:
                    stream = p.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=self.tts.sample_rate,  # Utiliser le sample rate de l'engine TTS
                        output=True,
                        frames_per_buffer=4096,
                    )

                    # Lire et jouer les chunks audio
                    while not self.abort_event.is_set():
                        try:
                            chunk = gen.audio_queue.get(timeout=0.1)
                            if chunk is None:  # Signal de fin
                                break
                            # Vérifier l'interruption avant d'écrire le chunk
                            if self.abort_event.is_set():
                                logger.debug("-- [INTERRUPTION] Stopping audio playback immediately")
                                break
                            stream.write(chunk)
                        except queue.Empty:
                            if gen.tts_completed:
                                break
                            continue

                    if not self.abort_event.is_set():
                        gen.audio_completed = True
                        logger.debug("-- [DONE] Audio")
                    else:
                        logger.debug("-- [STOP] Audio")

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
            logger.debug("-- [SHUTDOWN] Audio Player")

    def _abort_current(self):
        """Interrompt la génération en cours"""
        if not self.current_generation:
            return

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

    def start(self):
        """Démarre le système de conversation"""
        logger.debug("-- [START] Vocal Assistant")

        # Démarrer les workers
        self._start_workers()

        # Démarrer le STT (écoute du microphone)
        logger.debug("-- [START] STT Ready to Listen")
        self.stt.transcribe_loop()

    def shutdown(self):
        """Arrête proprement le système"""
        logger.debug("-- [SHUTDOWN] Vocal Assistant")

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
