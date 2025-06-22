import os
import yt_dlp

def download_youtube_audio(youtube_url: str, output_dir: str = "downloads"):
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get("id", None)
        title = info.get("title", None)
        audio_path = os.path.join(output_dir, f"{video_id}.mp3")

        print(f"✅ Downloaded: {title}")
        print(f"🎧 Audio saved to: {audio_path}")

    return audio_path, title
