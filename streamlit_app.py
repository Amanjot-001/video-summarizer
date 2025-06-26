import os
import uuid
import streamlit as st

from video.downloader import download_youtube_audio
from video.uploader import process_uploaded_video
from transcription.whisper_transcriber import transcribe_audio, save_transcript
from summarization.summarizer import (
    load_transcript,
    chunk_segments,
    get_local_hf_llm,
    summarize_chunks,
    save_summary,
)
from embeddings.embedder import build_faiss_index
from qna.qa_bot import load_qna_chain

# --- Streamlit Page Config ---
st.set_page_config(page_title="Video Summarizer", layout="centered")
st.title("🎬 Video Summarizer")
st.divider()

# --- Session State Defaults ---
for key, default in {
    "summary_text": "",
    "transcript_text": "",
    "qa_chain": None,
    "chunks": [],
    "audio_path": None,
    "title": "",
    "transcript_file": None,
    "summary_file": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Input Section ---
col1, col2 = st.columns([2, 1])
with col1:
    youtube_url = st.text_input("Paste YouTube Link", placeholder="https://youtube.com/...")
with col2:
    uploaded_file = st.file_uploader("Or Upload a Video", type=["mp4", "mkv", "mov"])

start_button = st.button("▶️ Start", use_container_width=True)

# --- Start Processing ---
if start_button:
    with st.spinner("Processing... Please wait..."):
        try:
            # Step 1: Download or Upload
            if youtube_url:
                audio_path, title = download_youtube_audio(youtube_url)
            elif uploaded_file:
                temp_path = f"uploads/{uuid.uuid4().hex[:8]}.mp4"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                audio_path, title = process_uploaded_video(temp_path)
            else:
                st.warning("❌ Please provide a YouTube URL or upload a video.")
                st.stop()

            st.session_state.audio_path = audio_path
            st.session_state.title = title or "video"

            # Step 2: Transcribe
            transcript, segments, language = transcribe_audio(audio_path)
            transcript_path = save_transcript(audio_path, transcript, segments, language)
            st.session_state.transcript_text = transcript
            st.session_state.transcript_file = transcript_path

            # Step 3: Chunk + Summarize
            chunks = chunk_segments(segments, max_chars=1000)
            st.session_state.chunks = chunks
            build_faiss_index(chunks)
            llm = get_local_hf_llm()
            summary = summarize_chunks(chunks, llm)
            st.session_state.summary_text = summary

            summary_path = f"summaries/{os.path.splitext(os.path.basename(audio_path))[0]}.txt"
            save_summary(summary, summary_path)
            st.session_state.summary_file = summary_path

            # Step 4: Load QA
            st.session_state.qa_chain = load_qna_chain()

            st.success("✅ Processing complete!")

        except Exception as e:
            st.error(f"❌ Error occurred: {e}")

# --- Summary Output ---
if st.session_state.summary_text:
    st.subheader("📄 Summary")
    st.text_area("Summary:", st.session_state.summary_text, height=200)
    st.download_button("⬇️ Download Summary", st.session_state.summary_text, file_name="summary.txt")

# --- Transcript Output ---
if st.session_state.transcript_text:
    st.subheader("🗣 Transcript")
    st.text_area("Transcript:", st.session_state.transcript_text[:3000] + " ...", height=200)
    st.download_button("⬇️ Download Transcript", st.session_state.transcript_text, file_name="transcript.txt")

# --- QnA Chatbot ---
if st.session_state.qa_chain:
    st.subheader("🤖 Ask Questions About the Video")
    user_question = st.text_input("Your Question:")
    if user_question:
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.qa_chain.run(user_question)
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"❌ Error in QnA: {e}")
