import json
import os
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_transcript(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["segments"]


def chunk_segments(segments, max_chars=1000):
    chunks = []
    current_chunk = []
    total_chars = 0

    for seg in segments:
        text = seg["text"]
        if total_chars + len(text) > max_chars:
            chunks.append(" ".join(s["text"] for s in current_chunk))
            current_chunk = []
            total_chars = 0
        current_chunk.append(seg)
        total_chars += len(text)

    if current_chunk:
        chunks.append(" ".join(s["text"] for s in current_chunk))
    return chunks


def get_local_hf_llm(model_name="facebook/bart-large-cnn"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=summarizer_pipeline)


def summarize_chunks(chunks, llm):
    docs = [Document(page_content=chunk) for chunk in chunks]

    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)
    summary = chain.invoke(docs)
    return summary['output_text'] if isinstance(summary, dict) else str(summary)


def save_summary(summary: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary saved to: {output_path}")
