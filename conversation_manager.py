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

    def __init__(self, user_text: str, gen_id: int):
        self.id = gen_id
        self.user_text = user_text
        self.assistant_text = ""
        self.text_queue = queue.Queue()

        # Flags LLM
        self.llm_completed = False
        self.llm_aborted = False

        # Flags TTS
        self.tts_started = False
        self.tts_completed = False
        self.tts_aborted = False

        # Flag d'interruption centrale (comme le projet original)
        self.abortion_started = False

        self.timestamp = time.time()


class ConversationManager:
    """
    Orchestre la conversation en temps réel avec 3 composants principaux:
    - STT (Speech-to-Text) avec Whisper
    - LLM (Large Language Model) avec Ollama
    - TTS (Text-to-Speech) avec EdgeTTS

    Architecture à 2 worker threads avec synchronisation robuste pour l'interruption.
    Basé sur le système d'interruption de RealtimeVoiceChat/code.
    """

    def __init__(
        self,
        llm_provider: str = "ollama",
        llm_model: str = "llama3.2:3b",
        system_prompt_file: str = "system_prompt.txt",
    ):
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
        self.tts = AudioProcessor()

        logger.debug("-- [Init] STT")
        self.stt = TranscriptionProcessor()

        # Connecter les callbacks STT
        self.stt.full_transcription_callback = self._on_user_input
        self.stt.on_recording_start_callback = self._on_user_interrupt

        # Turn detection
        if USE_TURN_DETECTION:
            logger.debug("Turn detection activé")

        # État de la conversation
        self.current_generation: Optional[Generation] = None
        self.history = []
        self._generation_counter = 0

        # --- Events de synchronisation (comme le projet original) ---
        self.shutdown_event = threading.Event()
        self.new_input_event = threading.Event()

        # Events pour signaler les requêtes d'arrêt aux workers
        self.stop_llm_request_event = threading.Event()
        self.stop_tts_request_event = threading.Event()

        # Events pour confirmer que les workers ont arrêté
        self.stop_llm_finished_event = threading.Event()
        self.stop_tts_finished_event = threading.Event()

        # Event de complétion d'abort
        self.abort_completed_event = threading.Event()
        self.abort_completed_event.set()  # Initialement pas d'abort en cours

        # Flags d'état des workers (pour savoir s'ils sont actifs)
        self.llm_generation_active = False
        self.tts_generation_active = False

        # Lock pour éviter les race conditions dans abort
        self._abort_lock = threading.Lock()

        # Worker threads
        self.llm_thread = None
        self.tts_thread = None

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

    def _on_user_input(self, text: str):
        """Callback appelé quand le STT a finalisé une transcription."""
        if not text or not text.strip():
            return

        text = text.strip()
        logger.info(f"--> USER: {text}")

        # Interrompre la génération en cours si elle existe
        self._abort_current(reason="new user input")

        # Créer une nouvelle génération
        self._generation_counter += 1
        self.current_generation = Generation(text, self._generation_counter)

        # Ajouter à l'historique
        self.history.append({"role": "user", "content": text})

        # Signaler le worker LLM
        self.new_input_event.set()

    def _on_user_interrupt(self):
        """Callback appelé quand l'utilisateur commence à parler (interruption)"""
        # Ne déclencher l'abort que si TTS est actif (comme le projet original)
        if self.tts_generation_active:
            logger.info("-- [INTERRUPTION] User speaking while TTS active")
            self._abort_current(reason="user interrupt during TTS")

    def _llm_worker(self):
        """
        Worker thread LLM avec vérification de stop_event.
        Basé sur _llm_inference_worker du projet original.
        """
        logger.debug("-- [START] LLM Worker")

        while not self.shutdown_event.is_set():
            # Attendre un nouvel input avec timeout
            ready = self.new_input_event.wait(timeout=1.0)
            if not ready:
                continue

            # Vérifier si un abort a été demandé pendant l'attente
            if self.stop_llm_request_event.is_set():
                logger.debug("-- [LLM] Abort détecté pendant l'attente")
                self.stop_llm_request_event.clear()
                self.stop_llm_finished_event.set()
                self.llm_generation_active = False
                continue

            self.new_input_event.clear()

            gen = self.current_generation
            if not gen or gen.abortion_started:
                logger.debug("-- [LLM] Pas de génération valide ou déjà abortée")
                continue

            gen_id = gen.id
            logger.debug(f"-- [LLM Gen {gen_id}] Démarrage génération")

            # Marquer le worker comme actif
            self.llm_generation_active = True
            self.stop_llm_finished_event.clear()

            try:
                for chunk in self.llm.generate(
                    text=gen.user_text,
                    history=self.history[:-1],
                    use_system_prompt=True,
                ):
                    # *** VÉRIFIER L'ABORT AVANT CHAQUE CHUNK (comme projet original) ***
                    if self.stop_llm_request_event.is_set():
                        logger.debug(f"-- [LLM Gen {gen_id}] Stop request détecté, arrêt")
                        self.stop_llm_request_event.clear()
                        gen.llm_aborted = True
                        break

                    # Vérifier aussi le flag d'abortion sur la génération
                    if gen.abortion_started:
                        logger.debug(f"-- [LLM Gen {gen_id}] Abortion flag détecté")
                        gen.llm_aborted = True
                        break

                    # Traiter le chunk normalement
                    gen.text_queue.put(chunk)
                    gen.assistant_text += chunk

                # Fin de la génération
                if not gen.llm_aborted:
                    gen.llm_completed = True
                    gen.text_queue.put(None)  # Signal de fin

                    # Ajouter la réponse complète à l'historique
                    self.history.append({"role": "assistant", "content": gen.assistant_text})
                    logger.info(f"--> Assistant: {gen.assistant_text}")

                    # Limiter la taille de l'historique
                    if len(self.history) > 20:
                        self.history = self.history[-20:]

            except Exception as e:
                logger.error(f"-- [LLM Gen {gen_id}] Erreur: {e}", exc_info=True)
                gen.llm_aborted = True
                gen.text_queue.put(None)

            finally:
                # Nettoyer l'état (toujours exécuté)
                self.llm_generation_active = False
                self.stop_llm_finished_event.set()  # Signaler que le worker a terminé

                if gen.llm_aborted:
                    logger.debug(f"-- [LLM Gen {gen_id}] Aborted, signaling TTS stop")
                    self.stop_tts_request_event.set()

                logger.debug(f"-- [LLM Gen {gen_id}] Worker cycle terminé")

    def _tts_worker(self):
        """
        Worker thread TTS avec vérification de stop_event.
        Basé sur _tts_quick_inference_worker du projet original.
        """
        logger.debug("-- [START] TTS Worker")

        while not self.shutdown_event.is_set():
            time.sleep(0.01)

            gen = self.current_generation
            if not gen or gen.tts_started:
                continue

            # Vérifier si abort demandé avant de commencer
            if self.stop_tts_request_event.is_set():
                logger.debug("-- [TTS] Abort détecté avant démarrage")
                self.stop_tts_request_event.clear()
                self.stop_tts_finished_event.set()
                self.tts_generation_active = False
                continue

            # Vérifier si la génération est déjà abortée
            if gen.abortion_started or gen.tts_aborted:
                logger.debug(f"-- [TTS Gen {gen.id}] Génération déjà abortée, skip")
                continue

            # Attendre qu'au moins un chunk soit disponible
            if gen.text_queue.empty():
                continue

            gen_id = gen.id
            gen.tts_started = True
            logger.debug(f"-- [TTS Gen {gen_id}] Démarrage synthèse")

            # Marquer le worker comme actif
            self.tts_generation_active = True
            self.stop_tts_finished_event.clear()

            # Générateur qui consomme la queue de texte avec vérification d'abort
            def text_chunks():
                while True:
                    # Vérifier l'abort dans le générateur
                    if self.stop_tts_request_event.is_set() or gen.abortion_started:
                        logger.debug(f"-- [TTS Gen {gen_id}] Stop détecté dans text_chunks")
                        break

                    try:
                        chunk = gen.text_queue.get(timeout=0.1)
                        if chunk is None:
                            break
                        yield chunk
                    except queue.Empty:
                        if gen.llm_completed or gen.llm_aborted:
                            break
                        continue

            try:
                # Vérifier une dernière fois avant de synthétiser
                if self.stop_tts_request_event.is_set() or gen.abortion_started:
                    logger.debug(f"-- [TTS Gen {gen_id}] Abort détecté avant synthèse")
                    gen.tts_aborted = True
                else:
                    # Synthétiser avec l'event de stop (passé au synthesizer)
                    completed = self.tts.synthesize_generator(
                        text_chunks(),
                        stop_event=self.stop_tts_request_event,
                    )

                    if completed and not self.stop_tts_request_event.is_set():
                        gen.tts_completed = True
                        logger.debug(f"-- [TTS Gen {gen_id}] Synthèse complète")
                    else:
                        gen.tts_aborted = True
                        logger.debug(f"-- [TTS Gen {gen_id}] Synthèse interrompue")

            except Exception as e:
                logger.error(f"-- [TTS Gen {gen_id}] Erreur: {e}", exc_info=True)
                gen.tts_aborted = True

            finally:
                # Nettoyer l'état (toujours exécuté)
                self.tts_generation_active = False
                self.stop_tts_finished_event.set()  # Signaler que le worker a terminé
                self.stop_tts_request_event.clear()

                logger.debug(f"-- [TTS Gen {gen_id}] Worker cycle terminé")

    def _abort_current(self, reason: str = ""):
        """
        Interrompt la génération en cours avec synchronisation robuste.
        Basé sur process_abort_generation du projet original.
        """
        with self._abort_lock:
            gen = self.current_generation
            gen_id_str = f"Gen {gen.id}" if gen else "Gen None"

            # Vérifier si abort nécessaire
            if gen is None or gen.abortion_started:
                if gen is None:
                    logger.debug(f"-- [ABORT] {gen_id_str} Pas de génération active")
                else:
                    logger.debug(f"-- [ABORT] {gen_id_str} Abortion déjà en cours")
                self.abort_completed_event.set()
                return

            # --- Démarrer le processus d'abort ---
            logger.info(f"-- [ABORT] {gen_id_str} Démarrage (reason: {reason})")
            gen.abortion_started = True
            self.abort_completed_event.clear()

            # --- Arrêter le LLM ---
            if self.llm_generation_active:
                logger.debug(f"-- [ABORT] {gen_id_str} Arrêt LLM...")
                self.stop_llm_request_event.set()
                self.new_input_event.set()  # Réveiller le worker s'il attend

                # Attendre confirmation avec timeout (comme projet original: 5s)
                stopped = self.stop_llm_finished_event.wait(timeout=5.0)
                if stopped:
                    logger.debug(f"-- [ABORT] {gen_id_str} LLM arrêté confirmé")
                    self.stop_llm_finished_event.clear()
                else:
                    logger.warning(f"-- [ABORT] {gen_id_str} Timeout attente arrêt LLM")

                # Appeler cancel_generation externe
                if hasattr(self.llm, "cancel_generation"):
                    try:
                        self.llm.cancel_generation()
                    except Exception as e:
                        logger.warning(f"-- [ABORT] {gen_id_str} Erreur cancel LLM: {e}")

                self.llm_generation_active = False
            else:
                logger.debug(f"-- [ABORT] {gen_id_str} LLM inactif, pas d'arrêt nécessaire")

            self.stop_llm_request_event.clear()

            # --- Arrêter le TTS ---
            if self.tts_generation_active:
                logger.debug(f"-- [ABORT] {gen_id_str} Arrêt TTS...")
                self.stop_tts_request_event.set()

                # Arrêter le stream TTS directement
                if hasattr(self.tts, "stream"):
                    try:
                        self.tts.stream.stop()
                    except Exception as e:
                        logger.warning(f"-- [ABORT] {gen_id_str} Erreur stop stream: {e}")

                # Attendre confirmation avec timeout (comme projet original: 5s)
                stopped = self.stop_tts_finished_event.wait(timeout=5.0)
                if stopped:
                    logger.debug(f"-- [ABORT] {gen_id_str} TTS arrêté confirmé")
                    self.stop_tts_finished_event.clear()
                else:
                    logger.warning(f"-- [ABORT] {gen_id_str} Timeout attente arrêt TTS")

                self.tts_generation_active = False
            else:
                logger.debug(f"-- [ABORT] {gen_id_str} TTS inactif, pas d'arrêt nécessaire")

            self.stop_tts_request_event.clear()

            # --- Vider la queue de texte ---
            if gen.text_queue:
                while not gen.text_queue.empty():
                    try:
                        gen.text_queue.get_nowait()
                    except queue.Empty:
                        break

            # --- Signaler la complétion ---
            logger.info(f"-- [ABORT] {gen_id_str} Terminé")
            self.abort_completed_event.set()

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

        # Abort toute génération en cours
        self._abort_current(reason="shutdown")

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
