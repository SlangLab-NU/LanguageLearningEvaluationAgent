# WhisperX Speaker Diarization

This repository contains scripts for performing speaker diarization on audio files using WhisperX, specifically to identify USER and NPC speakers in conversations.

## Prerequisites

Before using these scripts, you need to:

1. Install WhisperX and its dependencies:
   ```bash
   pip install whisperx
   ```

2. Get a Hugging Face token for accessing PyAnnote models:
   - Visit [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Accept the user agreement
   - Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

3. Set your Hugging Face token as an environment variable (optional):
   ```bash
   export HF_TOKEN=your_token_here
   ```

## Scripts

### 1. Speaker Diarization (`speaker_diarization.py`)

This script uses WhisperX to transcribe audio and perform speaker diarization, identifying different speakers in the audio file.

```bash
python speaker_diarization.py /path/to/audio.wav --hf_token your_token_here --min_speakers 2 --max_speakers 2
```

Options:
- `--output_dir`: Directory to save output files (default: same directory as input audio)
- `--model`: Whisper model to use (default: "large-v2")
- `--device`: Device to run inference on (default: "cuda" if available, else "cpu")
- `--compute_type`: Compute type for inference (default: "float16")
- `--batch_size`: Batch size for inference (default: 16)
- `--min_speakers`: Minimum number of speakers (default: 2)
- `--max_speakers`: Maximum number of speakers (default: 2)
- `--hf_token`: Hugging Face token for accessing PyAnnote models
- `--language`: Language code for transcription (default: auto-detect)

### 2. Extract Speaker Audio (`extract_speaker_audio.py`)

This script extracts audio segments for each speaker from the original audio file based on the diarization results.

```bash
python extract_speaker_audio.py /path/to/audio.wav /path/to/audio_diarization.txt
```

Options:
- `--output_dir`: Directory to save output audio files (default: same directory as input audio)
- `--user_speaker`: Speaker ID for USER (default: "SPEAKER_00")
- `--npc_speaker`: Speaker ID for NPC (default: "SPEAKER_01")

## Example Workflow

1. Run speaker diarization on your audio file:
   ```bash
   python speaker_diarization.py /home/osx/Documents/GitHub/LLEvalAgent/data/recordings_wav/P001-com.oculus.vrshell-20240807-093454.wav --hf_token your_token_here --min_speakers 2 --max_speakers 2
   ```

2. This will generate two files:
   - `P001-com.oculus.vrshell-20240807-093454_transcript.txt`: Transcription with speaker labels
   - `P001-com.oculus.vrshell-20240807-093454_diarization.txt`: Diarization segments

3. Extract audio for each speaker:
   ```bash
   python extract_speaker_audio.py /home/osx/Documents/GitHub/LLEvalAgent/data/recordings_wav/P001-com.oculus.vrshell-20240807-093454.wav /home/osx/Documents/GitHub/LLEvalAgent/data/recordings_wav/P001-com.oculus.vrshell-20240807-093454_diarization.txt
   ```

4. This will generate two audio files:
   - `P001-com.oculus.vrshell-20240807-093454_USER.wav`: Audio for USER
   - `P001-com.oculus.vrshell-20240807-093454_NPC.wav`: Audio for NPC

## Notes

- The speaker IDs (SPEAKER_00, SPEAKER_01, etc.) are assigned by the diarization model. You may need to adjust the `--user_speaker` and `--npc_speaker` options based on the actual speaker IDs in your diarization results.
- For better results, you can try different Whisper models (e.g., "medium", "large-v2", "large-v3").
- If you're running on a machine with limited GPU memory, try reducing the batch size or using a smaller model.
- For CPU-only machines, set `--device cpu` and `--compute_type int8`. 