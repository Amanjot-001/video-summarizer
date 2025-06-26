from transcription.whisper_transcriber import transcribe_audio, save_transcript
from video.downloader import download_youtube_audio
from video.uploader import process_uploaded_video
from summarization.summarizer import (
    load_transcript,
    chunk_segments,
    get_local_hf_llm,
    summarize_chunks,
    save_summary,
)
from embeddings.embedder import build_faiss_index
from qna.qa_bot import load_qna_chain
import os

# video_url = "https://www.youtube.com/watch?v=QUTYxwTsbiM" # simple english story
video_url = "https://www.youtube.com/watch?v=aMx0d42wzBs" # french

if __name__ == "__main__":    
    mode = "1";
    if mode == "1":
        youtube_url = video_url
        audio_path, title = download_youtube_audio(youtube_url)
    elif mode == "2":
        file_path = input("📁 Enter path to local video file (e.g., .mp4): ")
        audio_path, title = process_uploaded_video(file_path)
    else:
        raise ValueError("Invalid input. Choose 1 or 2.")

    transcript, segments, language = transcribe_audio(audio_path, model_size="small")
    transcript_path = save_transcript(audio_path, transcript, segments, language)

    segments = load_transcript(transcript_path)
    chunks = chunk_segments(segments, max_chars=1000)
    
    build_faiss_index(chunks)
    
    llm = get_local_hf_llm()
    summary = summarize_chunks(chunks, llm)
    save_summary(summary, f"summaries/{os.path.splitext(os.path.basename(audio_path))[0]}.txt")
    
    qa_chain = load_qna_chain()
    while True:
        question = input("\nAsk a question about the video (or 'exit'): ")
        if question.lower() == "exit":
            break
        answer = qa_chain.invoke({"question": question})
        print("💬", answer['answer'])
