#!/usr/bin/env python3
"""
Assistant Vocal en Temps Réel (Français)

Application de conversation vocale en temps réel utilisant:
- STT: Whisper tiny (transcription parole → texte)
- LLM: Ollama llama3.2:3b (génération de réponses)
- TTS: Kokoro voix française (synthèse texte → parole)

Usage:
    uv run main.py

Ctrl+C pour quitter
"""

import logging
import signal
import sys
import time

from conversation_manager import ConversationManager
from logsetup import setup_logging

# Configuration du logging
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

# Instance globale pour le signal handler
manager = None


def signal_handler(sig, frame):
    """Gestionnaire de signal pour arrêt propre (Ctrl+C)"""
    print("\n\n🛑 Arrêt en cours...")

    if manager:
        try:
            manager.shutdown()
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt: {e}")

    print("👋 Au revoir!")
    sys.exit(0)


def print_banner():
    """Affiche la bannière de démarrage"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     🎙️  ASSISTANT VOCAL TEMPS RÉEL (Français)                     ║
║                                                                  ║
║  ────────────────────────────────────────────────────────────    ║
║                                                                  ║
║  🎧  STT: Whisper tiny (français)                                ║
║  🧠  LLM: Ollama llama3.2:3b                                     ║
║  🔊  TTS: Kokoro (voix française)                                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_dependencies():
    """Vérifie que les dépendances système sont disponibles"""
    import subprocess

    logger.info("🔍 Vérification des dépendances...")

    # Vérifier Ollama
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if "llama3.2:3b" in result.stdout:
            logger.info("✅ Ollama llama3.2:3b trouvé")
        else:
            logger.warning(
                "⚠️ Ollama llama3.2:3b non trouvé. Téléchargez-le avec: ollama pull llama3.2:3b"
            )
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("❌ Ollama non trouvé. Installez-le depuis https://ollama.ai")
        return False

    logger.info("✅ Dépendances OK")
    return True


def main():
    """Point d'entrée principal"""
    global manager

    # Afficher la bannière
    print_banner()

    # Vérifier les dépendances
    if not check_dependencies():
        print("\n❌ Dépendances manquantes. Veuillez les installer.")
        print("\nPour Ollama:")
        print("  1. Installez depuis https://ollama.ai")
        print("  2. Téléchargez le modèle: ollama pull llama3.2:3b")
        sys.exit(1)

    # Configurer le gestionnaire de signal
    signal.signal(signal.SIGINT, signal_handler)

    print("\n⏳ Initialisation de l'assistant...")
    print("   (Cela peut prendre quelques secondes...)\n")

    try:
        # Créer le gestionnaire de conversation
        manager = ConversationManager(
            llm_provider="ollama",
            llm_model="llama3.2:3b",
            tts_engine="kokoro",
            system_prompt_file="system_prompt.txt",
        )

        # Démarrer l'assistant
        manager.start()

        print("\n" + "=" * 70)
        print("✅ Assistant prêt! Commencez à parler...")
        print("=" * 70)
        print("\n💡 Conseils d'utilisation:")
        print("   • Parlez clairement dans votre microphone")
        print("   • Attendez la fin de la réponse ou interrompez en parlant")
        print("   • Appuyez sur Ctrl+C pour quitter")
        print("\n" + "─" * 70 + "\n")

        # Boucle principale - garde l'application en vie
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            signal_handler(None, None)

    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}", exc_info=True)
        print(f"\n❌ Erreur: {e}")
        print("\nVérifiez les logs pour plus de détails.")
        sys.exit(1)


if __name__ == "__main__":
    main()
