# Assistant Vocal Temps Réel

Assistant vocal conversationnel en temps réel utilisant les technologies de pointe pour la reconnaissance vocale, la génération de langage et la synthèse vocale.

## Fonctionnalités

- Conversation vocale en temps réel avec reconnaissance et synthèse instantanées
- Support multilingue (français, anglais, et autres langues supportées par Whisper et EdgeTTS)
- Interruptions naturelles - vous pouvez interrompre l'assistant en parlant
- Turn detection intelligent - détection automatique des fins de phrases
- Streaming LLM - réponses fluides générées en temps réel
- Interface terminal simple - pas besoin de navigateur
- Architecture multi-thread optimisée pour une latence minimale

## Technologies

| Composant | Technologie | Description |
| --------- | ----------- | ----------- |
| **STT** | Whisper small | Reconnaissance vocale multilingue rapide et précise |
| **LLM** | Ollama ministral-3 | Génération de réponses intelligentes en streaming |
| **TTS** | EdgeTTS | Synthèse vocale multilingue via Azure (voix neuronales) |

## 📋 Prérequis

### Dépendances système

1. **Python 3.10+** avec `uv` installé
2. **Ollama** avec le modèle `ministral-3` ou `llama3.2:3b`
3. **PortAudio** pour l'accès au microphone
4. **FFmpeg** pour le traitement audio
5. **MPV** pour la lecture audio (utilisé par EdgeTTS)

### Installation macOS

```bash
# Installer Homebrew si nécessaire
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Installer les dépendances système
brew install portaudio ffmpeg mpv

# Installer Ollama
brew install ollama

# Démarrer Ollama et télécharger le modèle
ollama serve &
ollama pull ministral-3
# ou
ollama pull llama3.2:3b
```

### Installation Linux (Ubuntu/Debian)

```bash
# Dépendances système
sudo apt-get update
sudo apt-get install portaudio19-dev ffmpeg mpv

# Installer Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Télécharger le modèle
ollama pull ministral-3
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

1. Vérifier qu'Ollama et le modèle LLM sont disponibles
2. Initialiser les composants (STT, LLM, TTS)
3. Télécharger les modèles Whisper si nécessaire (~40 MB pour tiny, ~140 MB pour small)
4. Démarrer l'écoute du microphone

### Utilisation

1. **Parlez clairement** dans votre microphone
2. **Attendez** la transcription et la réponse
3. **Interrompez** l'assistant en recommençant à parler
4. **Quittez** avec `Ctrl+C`

## Configuration

### Changer la voix EdgeTTS

Éditez [tts_module.py](tts_module.py#L11) pour modifier la voix :

```python
EDGE_TTS_VOICE = "en-US-AvaMultilingualNeural"  # Voix multilingue par défaut
```

Voix populaires:

- **Multilingue**: `en-US-AvaMultilingualNeural` (supporte français, anglais, espagnol, etc.)
- **Français**: `fr-FR-DeniseNeural`, `fr-FR-HenriNeural`
- **Anglais**: `en-US-AriaNeural`, `en-GB-SoniaNeural`

Liste complète : [Microsoft TTS Voices](https://learn.microsoft.com/azure/ai-services/speech-service/language-support)

### Modifier le prompt système

Éditez [system_prompt.txt](system_prompt.txt) pour changer la personnalité de l'assistant.

### Ajuster le modèle Whisper

Éditez [stt_module.py](stt_module.py#L28) :

```python
"model": "small",  # Options: "tiny", "base", "small", "medium"
```

**Note**: Les modèles plus grands sont plus précis mais plus lents.

### Changer le modèle LLM

Éditez [main.py](main.py#L27) :

```python
LLM_MODEL = "ministral-3"  # Options: "ministral-3", "llama3.2:3b", etc.
```

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
│  - Multilingue      │
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
│  - MP3/Opus         │
│  - Lecture via MPV  │
└──────┬──────────────┘
       │ Audio
       ↓
┌─────────────┐
│ Haut-parleurs│
└─────────────┘
```

### Architecture Multi-Thread

Le système utilise **deux threads workers** indépendants pour minimiser la latence:

#### Thread 1: LLM Worker

- Écoute les nouvelles entrées utilisateur via `new_input_event`
- Génère les réponses en streaming via l'API Ollama
- Place chaque chunk de texte dans `text_queue`
- Gère l'interruption via `abort_event`

#### Thread 2: TTS Worker

- Consomme les chunks de texte de `text_queue`
- Synthétise l'audio via EdgeTTS
- **Joue l'audio directement via MPV** (format MP3/Opus)
- Supporte l'interruption pour les réponses réactives

**Note importante**: EdgeTTS génère de l'audio au format MP3/Opus qui est joué directement par MPV. Il n'y a pas de conversion en PCM ni de thread Audio Player séparé, ce qui simplifie l'architecture et améliore la performance.

### Flux de données

```text
USER INPUT → STT → [text_queue] → LLM → [text_queue] → TTS → MPV → SPEAKERS
                     ↑                                       ↑
                     └────── abort_event (interruption) ────┘
```

### Gestion des interruptions

Le système supporte deux types d'interruptions:

1. **Interruption par la voix**: Détectée par `stt_module` via `on_recording_start_callback`
2. **Détection de silence**: Gérée par `silence_active_callback` qui surveille l'état du silence

Quand une interruption est détectée:

- `abort_event` est activé
- Les deux threads arrêtent leur traitement en cours
- La queue `text_queue` est vidée
- Le stream audio MPV est arrêté
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
- **[conversation_manager.py](conversation_manager.py)** - Orchestration des 2 threads workers
- **[stt_module.py](stt_module.py)** - Reconnaissance vocale avec RealtimeSTT
- **[tts_module.py](tts_module.py)** - Synthèse vocale avec EdgeTTS
- **[llm_module.py](llm_module.py)** - Interface avec Ollama
- **[turn_detection.py](turn_detection.py)** - Détection intelligente des tours de parole
- **[text_similarity.py](text_similarity.py)** - Comparaison de textes pour déduplication

## 🐛 Dépannage

### "Ollama connection refused"

```bash
# Vérifier qu'Ollama tourne
ollama serve

# Dans un autre terminal
ollama list  # Doit afficher ministral-3 ou llama3.2:3b
```

### "Microphone not found"

- **macOS**: Vérifiez les permissions micro dans Préférences Système → Confidentialité
- **Linux**: Vérifiez que votre utilisateur est dans le groupe `audio`

### "EdgeTTS voice not found"

Vérifiez que la voix est correctement spécifiée dans [tts_module.py](tts_module.py#L11). Les voix EdgeTTS nécessitent une connexion Internet.

### "MPV not found"

EdgeTTS nécessite MPV pour jouer l'audio:

```bash
# macOS
brew install mpv

# Linux
sudo apt-get install mpv
```

### Audio robotique ou incompréhensible

Si l'audio EdgeTTS sonne mal, vérifiez que:

1. MPV est bien installé (`which mpv`)
2. Vous avez une connexion Internet active
3. La voix spécifiée existe (voir la liste Microsoft)

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
├── README.md                # Ce fichier (documentation)
├── main.py                  # Point d'entrée de l'application
├── conversation_manager.py  # Orchestrateur des 2 threads workers
├── stt_module.py            # Module STT (Whisper + RealtimeSTT)
├── tts_module.py            # Module TTS (EdgeTTS)
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

Modifiez le niveau de log dans [main.py](main.py#L39) :

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

# Test du module TTS
uv run python -c "from tts_module import AudioProcessor; tts = AudioProcessor(); print('TTS OK')"
```

## Améliorations futures

- Commandes vocales (stop, recommence, etc.)
- Historique persistant des conversations
- Choix de voix EdgeTTS via arguments CLI
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
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) et [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) par KoljaB - frameworks temps réel
