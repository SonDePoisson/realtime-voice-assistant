# Assistant Vocal Temps Réel

Assistant vocal conversationnel en temps réel utilisant les technologies de pointe pour la reconnaissance vocale, la génération de langage et la synthèse vocale.

## Fonctionnalités

- Conversation vocale en temps réel avec reconnaissance et synthèse instantanées
- Support complet du français et de l'anglais (transcription et synthèse)
- Interruptions naturelles - vous pouvez interrompre l'assistant en parlant
- Turn detection intelligent - détection automatique des fins de phrases
- Streaming LLM - réponses fluides générées en temps réel
- Interface terminal simple - pas besoin de navigateur
- Architecture multi-thread optimisée pour une latence minimale

## Technologies

| Composant | Technologie | Description |
| --------- | ----------- | ----------- |
| **STT** | Whisper small | Reconnaissance vocale rapide et précise |
| **LLM** | Ollama ministral-3 | Génération de réponses intelligentes |
| **TTS** | EdgeTTS | Synthèse vocale multilingue via Azure |

## 📋 Prérequis

### Dépendances système

1. **Python 3.10+** avec `uv` installé
2. **Ollama** avec le modèle `llama3.2:3b`
3. **PortAudio** pour l'accès au microphone
4. **FFmpeg** pour le traitement audio

### Installation macOS

```bash
# Installer Homebrew si nécessaire
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Installer les dépendances système
brew install portaudio ffmpeg

# Installer Ollama
brew install ollama

# Démarrer Ollama et télécharger le modèle
ollama serve &
ollama pull llama3.2:3b
```

### Installation Linux (Ubuntu/Debian)

```bash
# Dépendances système
sudo apt-get update
sudo apt-get install portaudio19-dev ffmpeg

# Installer Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Télécharger le modèle
ollama pull llama3.2:3b
```

## 🚀 Installation

```bash
# Cloner ou naviguer vers le projet
cd ~/Code/realtime-voice-assistant

# Les dépendances Python sont déjà configurées dans pyproject.toml
# uv les installera automatiquement lors du premier lancement
```

## ▶️ Utilisation

### Démarrage simple

```bash
# Démarrer l'assistant
uv run main.py
```

### Au premier lancement

Le système va :
1. Vérifier qu'Ollama et llama3.2:3b sont disponibles
2. Initialiser les composants (STT, LLM, TTS)
3. Télécharger les modèles Whisper si nécessaire (~40 MB pour tiny)
4. Démarrer l'écoute du microphone

### Utilisation

1. **Parlez clairement** dans votre microphone
2. **Attendez** la transcription et la réponse
3. **Interrompez** l'assistant en recommençant à parler
4. **Quittez** avec `Ctrl+C`

## Configuration

### Changer le moteur TTS

Le système supporte deux moteurs TTS. Éditez [main.py](main.py:28) :

```python
TTS_MODEL = "edge_tts"  # Options: "edge_tts", "kokoro"
```

### Changer la voix EdgeTTS

Éditez [tts_module.py](tts_module.py:33) pour modifier les voix par langue :

```python
EDGE_TTS_VOICES = {
    "fr": "fr-FR-DeniseNeural",  # Voix française
    "en": "en-US-AvaMultilingualNeural",  # Voix anglaise
}
```

Liste des voix disponibles : [Microsoft TTS Voices](https://learn.microsoft.com/azure/ai-services/speech-service/language-support)

### Modifier le prompt système

Éditez [system_prompt.txt](system_prompt.txt) pour changer la personnalité de l'assistant.

### Ajuster le modèle Whisper

Éditez [stt_module.py](stt_module.py:28) :

```python
"model": "tiny",  # Options: "tiny", "base", "small", "medium"
```

**Note**: Les modèles plus grands sont plus précis mais plus lents.

## Architecture

### Vue d'ensemble

Le système est conçu autour d'une architecture pipeline en temps réel avec trois composants principaux:

```text
┌─────────────┐
│ Microphone  │
└──────┬──────┘
       │ Audio 48kHz
       ↓
┌─────────────────────┐
│  STT (Whisper)      │
│  - Modèle: small    │
│  - Langue: fr/en    │
└──────┬──────────────┘
       │ Texte transcrit
       ↓
┌─────────────────────┐
│ LLM (Ollama)        │
│  - ministral-3      │
│  - Streaming        │
└──────┬──────────────┘
       │ Réponse (chunks)
       ↓
┌─────────────────────┐
│  TTS (EdgeTTS)      │
│  - Voice: Multi     │
│  - Streaming        │
└──────┬──────────────┘
       │ Audio
       ↓
┌─────────────┐
│ Haut-parleurs│
└─────────────┘
```

### Architecture Multi-Thread

Le système utilise trois threads workers indépendants pour minimiser la latence:

#### Thread 1: LLM Worker

- Écoute les nouvelles entrées utilisateur via `new_input_event`
- Génère les réponses en streaming via l'API Ollama
- Place chaque chunk de texte dans `text_queue`
- Gère l'interruption via `abort_event`

#### Thread 2: TTS Worker

- Consomme les chunks de texte de `text_queue`
- Synthétise l'audio via EdgeTTS ou Kokoro
- Place les chunks audio dans `audio_queue`
- Supporte l'interruption pour les réponses réactives

#### Thread 3: Audio Player Worker

- Lit les chunks audio de `audio_queue`
- Joue l'audio via PyAudio (format: PCM 16-bit, 24kHz mono)
- Bufferise 5 chunks minimum avant de commencer
- Arrêt immédiat sur interruption utilisateur

### Flux de données

```text
USER INPUT → STT → [text_queue] → LLM → [text_queue] → TTS → [audio_queue] → Audio Player → SPEAKERS
                     ↑                                              ↑
                     └──────── abort_event (interruption) ─────────┘
```

### Gestion des interruptions

Le système supporte deux types d'interruptions:

1. **Interruption par la voix**: Détectée par `stt_module` via `on_recording_start_callback`
2. **Détection de silence**: Gérée par `silence_active_callback` qui surveille l'état du silence

Quand une interruption est détectée:

- `abort_event` est activé
- Les trois threads arrêtent leur traitement en cours
- Les queues `text_queue` et `audio_queue` sont vidées
- Une nouvelle génération peut démarrer

### Turn Detection

Le module `turn_detection.py` calcule dynamiquement le temps d'attente optimal avant de finaliser une transcription, basé sur:

- La longueur du texte transcrit
- La présence de ponctuation finale
- La latence estimée du pipeline (LLM + TTS)

États du turn detection:

- **Cold**: Aucune activité vocale
- **Potential End**: Silence détecté après ponctuation
- **Hot**: Prêt à finaliser la transcription
- **Final**: Transcription finalisée et envoyée au LLM

### Composants principaux

- **[main.py](main.py)** - Point d'entrée et gestion du cycle de vie
- **[conversation_manager.py](conversation_manager.py)** - Orchestration des 3 threads workers
- **[stt_module.py](stt_module.py)** - Reconnaissance vocale avec RealtimeSTT
- **[tts_module.py](tts_module.py)** - Synthèse vocale avec EdgeTTS/Kokoro
- **[llm_module.py](llm_module.py)** - Interface avec Ollama
- **[turn_detection.py](turn_detection.py)** - Détection intelligente des tours de parole
- **[text_similarity.py](text_similarity.py)** - Comparaison de textes pour déduplication

## 🐛 Dépannage

### "Ollama connection refused"

```bash
# Vérifier qu'Ollama tourne
ollama serve

# Dans un autre terminal
ollama list  # Doit afficher llama3.2:3b
```

### "Microphone not found"

- **macOS**: Vérifiez les permissions micro dans Préférences Système → Confidentialité
- **Linux**: Vérifiez que votre utilisateur est dans le groupe `audio`

### "EdgeTTS voice not found"

Vérifiez que la voix est correctement spécifiée dans [tts_module.py](tts_module.py:33). Les voix EdgeTTS nécessitent une connexion Internet.

### Audio haché ou saccadé avec Kokoro

Si vous utilisez le moteur Kokoro, augmentez la taille des buffers dans la configuration du moteur.

## Performance

| Métrique | Valeur typique |
| -------- | -------------- |
| Latence STT | ~0.5-1s |
| Latence LLM (TTFT) | ~0.5-1s |
| Latence TTS | ~0.2-0.4s |
| **Latence totale** | **~1.5-2.5s** |

Mesures effectuées sur MacBook M1/M2 avec ministral-3 et EdgeTTS.

## Structure du projet

```text
realtime-voice-assistant/
├── PLAN.md                  # Plan d'implémentation détaillé
├── README.md                # Ce fichier (documentation)
├── main.py                  # Point d'entrée de l'application
├── conversation_manager.py  # Orchestrateur des 3 threads workers
├── stt_module.py            # Module STT (Whisper + RealtimeSTT)
├── tts_module.py            # Module TTS (EdgeTTS / Kokoro)
├── llm_module.py            # Module LLM (interface Ollama)
├── turn_detection.py        # Détection intelligente des tours de parole
├── text_similarity.py       # Comparaison et déduplication de textes
├── logsetup.py              # Configuration du système de logging
├── system_prompt.txt        # Prompt système de l'assistant
├── pyproject.toml           # Configuration uv + dépendances Python
└── .venv/                   # Environnement virtuel Python
```

## Développement

### Lancer en mode debug

Modifiez le niveau de log dans [main.py](main.py:39) :

```python
setup_logging(logging.DEBUG)  # Au lieu de logging.INFO
```

### Architecture des callbacks

Le système utilise des callbacks pour la communication entre modules:

- `full_transcription_callback`: STT → ConversationManager (texte finalisé)
- `on_recording_start_callback`: STT → ConversationManager (interruption détectée)
- `silence_active_callback`: STT → ConversationManager (état du silence)
- `on_first_audio_chunk_synthesize`: TTS → ConversationManager (premier chunk audio)

### Tests des composants

```bash
# Test du module LLM
uv run python llm_module.py

# Test du module TTS (nécessite EdgeTTS ou Kokoro)
uv run python -c "from tts_module import AudioProcessor; tts = AudioProcessor('edge_tts', 'fr'); print('TTS OK')"
```

## Améliorations futures

- Commandes vocales (stop, recommence, etc.)
- Historique persistant des conversations
- Choix de voix et moteur TTS via arguments CLI
- Support multilingue avec changement de langue en temps réel
- Métriques de latence et performance en temps réel
- Mode push-to-talk optionnel
- Interface web optionnelle pour monitoring

## Licence

Ce projet est basé sur le projet [RealtimeVoiceChat](https://github.com/KoljaB/RealtimeVoiceChat) et utilise les bibliothèques open-source suivantes:

- RealtimeSTT (MIT)
- RealtimeTTS (MIT)
- Transformers (Apache 2.0)
- Ollama (MIT)
- edge-tts (GPL-3.0)

## Remerciements

- [Whisper](https://github.com/openai/whisper) par OpenAI - reconnaissance vocale de haute qualité
- [Ollama](https://ollama.ai) - inférence LLM locale optimisée
- [EdgeTTS](https://github.com/rany2/edge-tts) - synthèse vocale via Microsoft Azure
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - synthèse vocale multilingue alternative
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) et [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) par KoljaB - frameworks temps réel
