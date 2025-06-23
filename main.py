from transcription.whisper_transcriber import transcribe_audio, save_transcript
from video.downloader import download_youtube_audio

# video_url = "https://www.youtube.com/watch?v=EQsQeBsB6YI"
video_url = "https://www.youtube.com/watch?v=QUTYxwTsbiM"

if __name__ == "__main__":
    youtube_url = video_url
    audio_path, title = download_youtube_audio(youtube_url)

    transcript, segments, language = transcribe_audio(audio_path, model_size="tiny")
    save_transcript(audio_path, transcript, segments, language)
    
