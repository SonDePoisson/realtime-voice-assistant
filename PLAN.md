# Plan d'implémentation: Terminal Voice Assistant (Français)

## Vue d'ensemble
Créer un assistant vocal en temps réel dans `~/Code/realtime-voice-assistant` avec conversation fluide, interruptions, et support français complet.

## Architecture Simplifiée

### Système Actuel (WebSocket, 4 threads):
- Thread 1: Request processor
- Thread 2: LLM inference
- Thread 3: TTS "quick answer"
- Thread 4: TTS "final answer"

### Nouveau Système (Terminal, 3 threads):
- Thread 1: STT (intégré dans RealtimeSTT)
- Thread 2: LLM inference
- Thread 3: TTS streaming unique

**Simplification clé**: Un seul TTS streaming au lieu de deux (quick+final), car l'utilisateur terminal n'a pas besoin d'optimisation sub-seconde.

## Configuration

### STT (RealtimeSTT)
- **Modèle**: `tiny` (au lieu de `base.en`)
- **Langue**: `fr` (au lieu de `en`)
- **Microphone**: `True` (au lieu de `False`)
- Garder: VAD (Silero), beam_size=3, tous les callbacks

### TTS (Kokoro)
- **Voix**: `af_sky` ou `af_bella` (français, au lieu de `af_heart` anglais)
- **Vitesse**: 1.26 (garder)
- **Lecture**: Directe vers haut-parleurs (pas de WebSocket)

### LLM (Ollama)
- **Modèle**: `llama3.2:3b` (déjà téléchargé)
- **Backend**: `ollama`
- **URL**: `http://127.0.0.1:11434`
- Aucun changement nécessaire

## Structure du Projet

```
~/Code/realtime-voice-assistant/
├── main.py                    # Point d'entrée terminal
├── conversation_manager.py    # Orchestrateur 3-threads (NOUVEAU)
├── stt_module.py             # Adapté de transcribe.py
├── tts_module.py             # Adapté de audio_module.py
├── llm_module.py             # Copié tel quel
├── turn_detection.py         # Copié tel quel (renommé depuis turndetect.py)
├── text_context.py           # Copié tel quel
├── text_similarity.py        # Copié tel quel
├── colors.py                 # Copié tel quel
├── logsetup.py               # Copié tel quel
├── system_prompt.txt         # Prompt système (français)
├── pyproject.toml            # Config uv avec dépendances
└── .venv/                    # Python 3.10
```

## Dépendances (via uv)

```toml
[project.dependencies]
realtimestt = "==0.3.104"
realtimetts = {extras = ["kokoro"], version = ">=0.5.5"}
requests = ">=2.32.5"
transformers = "*"
torch = "*"
numpy = "*"
scipy = "*"
nltk = "*"
sentence-transformers = "*"
```

## Plan d'Implémentation (8 étapes)

### ✅ Étape 1: Configuration Projet
- Créer répertoire ~/Code/realtime-voice-assistant
- Initialiser avec `uv init --python 3.10`
- Écrire ce plan dans PLAN.md

### Étape 2: Copier Fichiers Inchangés
Copier directement depuis RealtimeVoiceChat/code/:
- llm_module.py
- turndetect.py → turn_detection.py
- text_context.py
- text_similarity.py
- colors.py
- logsetup.py

### Étape 3: Configurer Dépendances
- Mettre à jour pyproject.toml avec les dépendances
- Installer: `uv sync`

### Étape 4: Adapter stt_module.py
Copier `transcribe.py` → `stt_module.py`

**Modifications clés**:
- `use_microphone`: True (était False)
- `model`: "tiny" (était "base.en")
- `language`: "fr" (était "en")
- `spinner`: True (feedback terminal)

### Étape 5: Adapter tts_module.py
Copier `audio_module.py` → `tts_module.py`

**Modifications clés**:
- `voice`: "af_sky" (était "af_heart")
- `muted`: False (lecture directe haut-parleurs)

### Étape 6: Créer conversation_manager.py
Nouveau fichier inspiré de speech_pipeline_manager.py
- 2 worker threads (LLM + TTS)
- Gestion interruptions
- Queue pour streaming texte

### Étape 7: Créer main.py
Point d'entrée terminal avec:
- Signal handlers (Ctrl+C)
- Initialisation ConversationManager
- Keep-alive loop

### Étape 8: Créer system_prompt.txt
Prompt système en français pour conversation vocale naturelle

## Points Critiques

### 1. Interruptions
- Déclencheur: `on_recording_start` du STT
- Actions: cancel LLM, stop TTS, vider queue

### 2. Turn Detection
- Garde callback `on_new_waiting_time`
- Met à jour durée silence dynamiquement
- Fonctionne cross-langue

### 3. Streaming Responsif
- Utiliser `TextContext.get_context()`
- Détecter première phrase complète
- Commencer TTS dès que possible

### 4. System Prompt
```
Tu es un assistant vocal français serviable et amical.
Réponds de manière concise et naturelle.
Tes réponses sont destinées à être lues à voix haute.
Évite les listes à puces ou formatage complexe.
Reste conversationnel et engageant.
```

## Tests

### Test 1: STT seul
```bash
uv run python -c "from stt_module import STTProcessor; ..."
```

### Test 2: TTS seul
```bash
uv run python -c "from tts_module import TTSProcessor; ..."
```

### Test 3: Pipeline complet
```bash
uv run main.py
```

## Dépendances Système

Vérifier:
```bash
ollama list | grep llama3.2:3b  # LLM
brew install portaudio ffmpeg  # Audio (macOS)
```
