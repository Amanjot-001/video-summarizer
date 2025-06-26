import os
import subprocess
import uuid

def process_uploaded_video(video_path: str, output_dir: str = "uploads"):
    os.makedirs(output_dir, exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    title = f"{base_name}_{unique_id}"
    audio_path = os.path.join(output_dir, f"{title}.mp3")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # disable video
        "-acodec", "libmp3lame",
        "-ar", "44100",
        "-ab", "192k",
        "-f", "mp3",
        audio_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Extracted audio from uploaded file.")
        print(f"🎧 Audio saved to: {audio_path}")
        return audio_path, title
    except subprocess.CalledProcessError:
        raise RuntimeError("❌ Failed to extract audio from uploaded video.")
