import pickle
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from helper import get_embedding
from unstructured.partition.pdf import partition_pdf
from get_chunks import process_images_with_captions, process_table_with_captions, process_raw_text
from langchain_community.embeddings.ollama import OllamaEmbeddings
import numpy as np
import os


def load_documents_data(load_path="faiss_index"):
    """Load documents data for BM25 search"""
    try:
        with open(f"{load_path}/documents_data.pkl", "rb") as f:
            documents_data = pickle.load(f)
        print(f"Documents data loaded from {load_path}/documents_data.pkl")
        return documents_data
    except Exception as e:
        print(f"Error loading documents data: {str(e)}")
        return None 


def prepare_documents_for_faiss(chunks):
    documents = []
    
    for i, chunk in enumerate(chunks):
        try:
            if not chunk.get("content"):
                continue
            
            metadata = {
                "chunk_id": i,
                "content_type": chunk.get("content_type", "text"),
                "filename": chunk.get("filename", ""),
                "caption": chunk.get("caption", ""),
                "image_text": chunk.get("image_text", ""),
            }
            
            if chunk.get("content_type") == "image" and "base64_image" in chunk:
                metadata["has_image"] = True
                metadata["page_number"] = chunk.get("page_number", "")
            
            if chunk.get("content_type") == "table" and "table_as_html" in chunk:
                metadata["has_table"] = True
                metadata["table_text"] = chunk.get("table_text", "")
            
            doc = Document(
                page_content=chunk["content"],
                metadata=metadata
            )
            
            documents.append(doc)
            
        except Exception as e:
            print(f"Error preparing document {i}: {str(e)}")
            continue
    
    print(f"Prepared {len(documents)} documents for FAISS")
    return documents


def create_faiss_index(documents, embedding_model="nomic-embed-text"):
    try:
        print("Creating FAISS index with embeddings...")
        embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url="http://localhost:11434"
        )
        
        vector_store = FAISS.from_documents(documents, embeddings)
        print(f"Successfully created FAISS index with {len(documents)} documents")
        return vector_store
    except Exception as e:
        print(f"Error creating FAISS index: {str(e)}")
        raise


def save_faiss_index(vector_store, save_path="faiss_index"):
    try:
        vector_store.save_local(save_path)
        print(f"FAISS index saved to {save_path}")
    except Exception as e:
        print(f"Error saving FAISS index: {str(e)}")
        raise


def save_documents_data(documents, save_path="faiss_index"):
    try:
        # Simply save the documents for BM25 to use
        documents_data = {
            'documents': documents
        }
        
        with open(f"{save_path}/documents_data.pkl", "wb") as f:
            pickle.dump(documents_data, f)
        
        print(f"Documents data saved to {save_path}/documents_data.pkl")
        
    except Exception as e:
        print(f"Error saving documents data: {str(e)}")
        raise


def load_faiss_index(load_path="faiss_index", embedding_model="nomic-embed-text"):
    try:
        embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url="http://localhost:11434"
        )
        
        vector_store = FAISS.load_local(
            load_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"FAISS index loaded from {load_path}")
        return vector_store
    except Exception as e:
        print(f"Error loading FAISS index: {str(e)}")
        raise


def ingest_all_content_into_faiss(processed_images, processed_tables, semantic_chunks, save_path="faiss_index"):
    all_chunks = processed_images + processed_tables + semantic_chunks
    print(f"Total chunks for FAISS ingestion: {len(all_chunks)}")
    
    documents = prepare_documents_for_faiss(all_chunks)
    
    vector_store = create_faiss_index(documents)
    
    save_faiss_index(vector_store, save_path)
    
    save_documents_data(documents, save_path)
    
    return vector_store


if __name__ == "__main__":
    from unstructured.partition.auto import partition
    
    pdf_path = "files/Attention Is All You Need.pdf"
    
    raw_chunks = partition(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Figure", "Table", "Image"],
        extract_image_block_to_payload=True,
        chunking_strategy=None,
    )
    

    
    processed_images = process_images_with_captions(raw_chunks, llm=True)
    processed_tables, table_errors = process_table_with_captions(raw_chunks, ollama=True)
    
    chunks = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            chunking_strategy="by_title",
            max_characters=2000,
            min_chars_to_combine=500,
            chars_before_new_chunk=1500,
        )
    semantic_chunks = process_raw_text(chunks, llm=False)
    
    
    # Ingest into FAISS
    vector_store = ingest_all_content_into_faiss(
        processed_images, 
        processed_tables, 
        semantic_chunks,
        save_path="faiss_index"
    )