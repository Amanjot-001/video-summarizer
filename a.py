import json
import os
import torch
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_huggingface import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate

def load_transcript(json_path: str) -> List[Dict[str, Any]]:
    """Load transcript segments from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", [])

def chunk_segments(segments: List[Dict[str, Any]], max_chars: int = 1000) -> List[str]:
    """Improved chunking that respects sentence boundaries."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    full_text = " ".join(seg["text"] for seg in segments)
    return text_splitter.split_text(full_text)

def get_local_hf_llm(model_name: str = "facebook/bart-large-cnn") -> HuggingFacePipeline:
    """Create HuggingFace pipeline with proper configuration."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    summarizer_pipeline = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        max_length=150,
        min_length=30,
        truncation=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    return HuggingFacePipeline(pipeline=summarizer_pipeline)

def create_summarization_chain(llm: HuggingFacePipeline) -> MapReduceDocumentsChain:
    """Create a proper map-reduce summarization chain with custom prompts."""
    # Map prompt
    map_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce prompt
    reduce_template = """Combine these summaries into one coherent summary:
    {text}
    COHERENT FINAL SUMMARY:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Combiner
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name="text"
    )

    # Reducer
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,
    )

    # Final chain
    return MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="text",
        return_intermediate_steps=False,
    )

def summarize_transcript(
    transcript_path: str,
    output_path: str,
    model_name: str = "facebook/bart-large-cnn"
) -> None:
    """Main function to process and summarize a transcript."""
    # Load and prepare data
    segments = load_transcript(transcript_path)
    chunks = chunk_segments(segments)
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    # Initialize model and chain
    llm = get_local_hf_llm(model_name)
    chain = create_summarization_chain(llm)
    
    # Execute summarization
    result = chain.invoke(docs)
    final_summary = result["output_text"]
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_summary)
    
    print(f"Summary successfully saved to: {output_path}")
    return final_summary

# Example usage
if __name__ == "__main__":
    summary = summarize_transcript(
        transcript_path="path/to/your/transcript.json",
        output_path="output/summary.txt",
        model_name="facebook/bart-large-cnn"
    )
    print("\nGenerated Summary:\n", summary)