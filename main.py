#!/usr/bin/env python3
"""
Assistant Vocal en Temps Réel

Application de conversation vocale en temps réel utilisant:
- STT: Whisper small (transcription parole → texte)
- LLM: Ollama llama3.2:3b (génération de réponses)
- TTS: Kokoro voix française (synthèse texte → parole)

Usage:
    uv run main.py

Ctrl+C pour quitter
"""

import os
import logging
import signal
import sys
import time
import warnings

from conversation_manager import ConversationManager
from logsetup import setup_logging

LLM_PROVIDER = "ollama"
LLM_MODEL = "llama3.2:3b"
TTS_MODEL = "kokoro"
STT_MODEL = "small"  # Configure in stt_module
LANGUAGE = "en"

# Désactiver les warnings pour un affichage plus propre
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Éviter les deadlocks avec HuggingFace tokenizers
warnings.filterwarnings("ignore", category=UserWarning)  # Warnings PyTorch/Whisper
warnings.filterwarnings("ignore", category=FutureWarning)  # Warnings PyTorch
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Warnings numpy/whisper

# Configuration du logging
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)


manager = None


def signal_handler(sig, frame):
    """Gestionnaire de signal pour arrêt propre (Ctrl+C)"""
    print("\n\nArrêt en cours...")

    if manager:
        try:
            manager.shutdown()
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt: {e}")

    sys.exit(0)


def check_dependencies():
    """Vérifie que les dépendances système sont disponibles"""
    import subprocess

    logger.debug("Vérification des dépendances...")

    # Vérifier Ollama
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if "llama3.2:3b" in result.stdout:
            logger.debug("Ollama llama3.2:3b trouvé")
        else:
            logger.warning("Ollama llama3.2:3b non trouvé. Téléchargez-le avec: ollama pull llama3.2:3b")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("Ollama non trouvé. Installez-le depuis https://ollama.ai")
        return False

    logger.debug("Dépendances OK")
    return True


def main():
    """Point d'entrée principal"""
    global manager

    # Vérifier les dépendances
    if not check_dependencies():
        print("\nDépendances manquantes. Veuillez les installer.")
        print("\nPour Ollama:")
        print("  1. Installez depuis https://ollama.ai")
        print("  2. Téléchargez le modèle: ollama pull llama3.2:3b")
        sys.exit(1)

    # Configurer le gestionnaire de signal
    signal.signal(signal.SIGINT, signal_handler)

    print("\nInitialisation de l'assistant...")
    print("(Cela peut prendre quelques secondes...)\n")

    try:
        # Créer le gestionnaire de conversation
        manager = ConversationManager(
            llm_provider=LLM_PROVIDER,
            llm_model=LLM_MODEL,
            tts_engine=TTS_MODEL,
            language=LANGUAGE,
            system_prompt_file="system_prompt.txt",
        )

        # Démarrer l'assistant
        manager.start()

        # Boucle principale - garde l'application en vie
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            signal_handler(None, None)

    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)
        print(f"\nErreur: {e}")
        print("\nVérifiez les logs pour plus de détails.")
        sys.exit(1)


if __name__ == "__main__":
    main()
