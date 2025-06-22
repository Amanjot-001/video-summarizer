from video.downloader import download_youtube_audio

video_url = "https://www.youtube.com/watch?v=EQsQeBsB6YI"

if __name__ == "__main__":
    youtube_url = video_url
    audio_path, title = download_youtube_audio(youtube_url)
