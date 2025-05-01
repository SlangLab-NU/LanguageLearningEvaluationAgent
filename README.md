# ELLMat: English Language Learning Multimodal Assessment Tool

This repository contains tools for processing and evaluating English language learning videos using multimodal analysis. The project offers two main approaches for assessment:

## Approach 1: Comprehensive Multimodal Evaluation

This approach provides a detailed analysis of various aspects of language proficiency using multiple evaluators.

### Project Structure

- `data/`: Contains raw and processed recordings and transcripts
  - `recordings/`: Original video recordings
  - `recordings_wav/`: Converted WAV audio files
  - `recordings_wav_processed/`: Processed audio files
  - `sample_transcripts/`: Example transcripts

- `process_recording/`: Tools for audio/video processing
  - Speaker diarization
  - Audio extraction
  - WAV conversion
  - Batch processing utilities

- `evaluation/`: Evaluation scripts and metrics
  - Text-based evaluation
  - Speech analysis
  - Overall score calculation

- `evaluator/`: Different types of evaluators
  - Base evaluator framework
  - Specialized evaluators for different aspects

- `tests/`: Unit tests for evaluators
  - Tests for vocabulary, grammar, fluency, etc.

- `utils/`: Utility functions and LLM integration

### Data Processing Pipeline

1. **Video Processing**
   ```bash
   cd process_recording
   # Convert videos to WAV format
   ./convert_to_wav.sh
   
   # Perform speaker diarization
   ./speaker_diarization.sh
   
   # Extract speaker audio
   ./extract_speaker_audio.sh
   
   # Extract transcripts
   ./extract_user_transcription.sh
   ```

2. **Running Evaluations**
   ```bash
   cd evaluation
   
   # Run text evaluation
   ./run_text_evaluation.sh
   
   # Run fluency evaluation
   ./run_fluency_evaluation.sh
   
   # Calculate overall scores
   ./overall_score_weighted.sh
   ```

## Approach 2: CEFR Level Prediction

This approach uses the CEFR-English-Level-Predictor to assess English proficiency levels.

### Using CEFR Predictor

1. **Setup**
   ```bash
   cd CEFR-English-Level-Predictor
   pip install -r requirements.txt
   python setup.py install
   ```

2. **Batch Processing**
   ```bash
   python predict_cefr_batch.py --input /path/to/transcripts --output results
   ```

   Options:
   - `--use-plus-levels`: Enable plus-level system (e.g., B1+, B2+)

### Output Format

The CEFR predictor generates two files:
- `cefr_scores.json`: Detailed results for each participant
- `cefr_summary.txt`: Human-readable summary of CEFR levels

## Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`
- FFmpeg for audio processing
- Whisper for speech recognition
- GPU recommended for faster processing

## Getting Started

1. Clone the repository
2. Install dependencies
3. Place your video recordings in `data/recordings/`
4. Choose your preferred evaluation approach:
   - For comprehensive analysis: Follow Approach 1
   - For CEFR level prediction only: Follow Approach 2

## Results

Results will be stored in:
- `evaluation/results/` for Approach 1
- `CEFR-English-Level-Predictor/results/` for Approach 2

## Notes

- Ensure sufficient disk space for audio processing
- GPU is recommended for faster processing
- Keep original videos backed up before processing
