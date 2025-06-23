# transcription/whisper_transcriber.py

import whisper
import json
import os

def transcribe_audio(audio_path: str, model_size: str = "base"):
    print("Loading Whisper model...")
    model = whisper.load_model(model_size)

    print("Transcribing audio...")
    result = model.transcribe(audio_path)

    transcript = result['text']
    segments = result['segments']
    language = result['language']

    print("Transcription complete.")
    return transcript, segments, language

def save_transcript(audio_path: str, transcript: str, segments: list, language: str, output_dir: str = "transcripts"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.json")

    data = {
        "language": language,
        "text": transcript,
        "segments": segments,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"📄 Transcript saved to: {output_path}")
    return output_path
