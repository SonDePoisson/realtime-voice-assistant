# 🎙️ Assistant Vocal Temps Réel (Français)

Assistant vocal conversationnel en temps réel utilisant les technologies de pointe pour la reconnaissance vocale, la génération de langage et la synthèse vocale.

## 🌟 Fonctionnalités

- **Conversation vocale en temps réel** avec reconnaissance et synthèse instantanées
- **Support complet du français** (transcription et synthèse)
- **Interruptions naturelles** - vous pouvez interrompre l'assistant en parlant
- **Turn detection intelligent** - détection automatique des fins de phrases
- **Streaming LLM** - réponses fluides générées en temps réel
- **Interface terminal simple** - pas besoin de navigateur

## 🛠️ Technologies

| Composant | Technologie | Description |
|-----------|------------|-------------|
| **STT** | Whisper tiny | Reconnaissance vocale rapide et précise |
| **LLM** | Ollama llama3.2:3b | Génération de réponses intelligentes |
| **TTS** | Kokoro (voix française) | Synthèse vocale naturelle |

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

## ⚙️ Configuration

### Changer la voix française

Éditez [tts_module.py](tts_module.py:123) :

```python
self.engine = KokoroEngine(
    voice="af_sky",  # Options: "af_sky", "af_bella", "af"
    ...
)
```

### Modifier le prompt système

Éditez [system_prompt.txt](system_prompt.txt) pour changer la personnalité de l'assistant.

### Ajuster le modèle Whisper

Éditez [stt_module.py](stt_module.py:28) :

```python
"model": "tiny",  # Options: "tiny", "base", "small", "medium"
```

**Note**: Les modèles plus grands sont plus précis mais plus lents.

## 🏗️ Architecture

```
┌─────────────┐
│ Microphone  │
└──────┬──────┘
       │ Audio 48kHz
       ↓
┌─────────────────────┐
│  STT (Whisper)      │
│  - Modèle: tiny     │
│  - Langue: fr       │
└──────┬──────────────┘
       │ Texte transcrit
       ↓
┌─────────────────────┐
│ LLM (Ollama)        │
│  - llama3.2:3b      │
│  - Streaming        │
└──────┬──────────────┘
       │ Réponse (chunks)
       ↓
┌─────────────────────┐
│  TTS (Kokoro)       │
│  - Voix: af_sky     │
│  - Français         │
└──────┬──────────────┘
       │ Audio
       ↓
┌─────────────┐
│ Haut-parleurs│
└─────────────┘
```

### Composants principaux

- **[main.py](main.py)** - Point d'entrée et gestion du cycle de vie
- **[conversation_manager.py](conversation_manager.py)** - Orchestration des 3 composants
- **[stt_module.py](stt_module.py)** - Reconnaissance vocale avec RealtimeSTT
- **[tts_module.py](tts_module.py)** - Synthèse vocale avec RealtimeTTS
- **[llm_module.py](llm_module.py)** - Interface avec Ollama
- **[turn_detection.py](turn_detection.py)** - Détection intelligente des tours de parole

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

### "Kokoro voice af_sky not found"

Essayez une autre voix française :
- `af_bella`
- `af` (voix générique)

### Audio haché ou saccadé

Augmentez la taille des chunks dans [tts_module.py](tts_module.py:102) :

```python
self.current_stream_chunk_size = 30  # Augmenter de 8 à 30
```

## 📊 Performance

| Métrique | Valeur typique |
|----------|----------------|
| Latence STT | ~0.5-1s |
| Latence LLM (TTFT) | ~1-2s |
| Latence TTS | ~0.3-0.5s |
| **Latence totale** | **~2-3s** |

*Mesures sur MacBook M1/M2 avec llama3.2:3b*

## 📝 Structure du projet

```
realtime-voice-assistant/
├── PLAN.md                  # Plan d'implémentation détaillé
├── README.md                # Ce fichier
├── main.py                  # Point d'entrée
├── conversation_manager.py  # Orchestrateur principal
├── stt_module.py           # Module STT (Whisper)
├── tts_module.py           # Module TTS (Kokoro)
├── llm_module.py           # Module LLM (Ollama)
├── turn_detection.py       # Détection des tours de parole
├── text_context.py         # Analyse de contexte textuel
├── text_similarity.py      # Similarité de textes
├── colors.py               # Utilitaires couleurs terminal
├── logsetup.py             # Configuration du logging
├── system_prompt.txt       # Prompt système de l'assistant
├── pyproject.toml          # Configuration uv + dépendances
└── .venv/                  # Environnement virtuel Python
```

## 🔧 Développement

### Lancer en mode debug

```bash
# Modifier le niveau de log dans main.py
setup_logging(logging.DEBUG)
```

### Tester un composant isolément

```python
# Test STT
uv run python -c "from stt_module import TranscriptionProcessor; ..."

# Test TTS
uv run python -c "from tts_module import AudioProcessor; ..."
```

## 🎯 Améliorations futures

- [ ] Commandes vocales ("stop", "recommence")
- [ ] Historique persistant des conversations
- [ ] Choix de voix via arguments CLI
- [ ] Indicateur visuel d'activité (animation terminal)
- [ ] Support multilingue (en/fr switchable)
- [ ] Métriques de latence en temps réel
- [ ] Mode "écoute continue" vs "push-to-talk"

## 📜 Licence

Ce projet est basé sur le projet [RealtimeVoiceChat](https://github.com/KoljaB/RealtimeVoiceChat) et utilise les bibliothèques open-source suivantes:
- RealtimeSTT (MIT)
- RealtimeTTS (MIT)
- Transformers (Apache 2.0)
- Ollama (MIT)

## 🙏 Remerciements

- [Whisper](https://github.com/openai/whisper) par OpenAI
- [Ollama](https://ollama.ai) pour l'inférence LLM locale
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) pour la synthèse vocale française
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) et [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) par KoljaB
