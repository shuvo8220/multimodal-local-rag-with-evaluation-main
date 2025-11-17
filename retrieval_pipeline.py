from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from helper import get_embedding
from langchain_community.embeddings.ollama import OllamaEmbeddings
import numpy as np
import os
from rank_bm25 import BM25Okapi
import re
from ingesion_pipeline import load_faiss_index, load_documents_data


def preprocess_text(text):
    """Simple text preprocessing for BM25"""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = [token for token in text.split() if token]
    return tokens


def create_bm25_index(documents):
    """Create BM25 index from documents"""
    try:
        tokenized_docs = []
        for doc in documents:
            tokens = preprocess_text(doc.page_content)
            tokenized_docs.append(tokens)
            
        bm25 = BM25Okapi(tokenized_docs)
        return bm25
    except Exception as e:
        print(f"Error creating BM25 index: {e}")
        return None


def keyword_search(query_text, top_k=20):
    """
    Perform keyword search using BM25.
    Args:
        query_text (str): The query text to search for
        top_k (int): Number of results to return
    Returns:
        list: Search results in OpenSearch-like format
    """
    try:
        documents_data = load_documents_data("faiss_index")
        if not documents_data:
            return []
        
        documents = documents_data['documents']

        bm25 = create_bm25_index(documents)
        if not bm25:
            return []

        query_tokens = preprocess_text(query_text)

        bm25_scores = bm25.get_scores(query_tokens)

        top_indices = bm25_scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0.1:  # Filter very low scores
                doc = documents[idx]
                result = {
                    "_score": float(bm25_scores[idx]),
                    "_source": {
                        "content": doc.page_content,
                        "content_type": doc.metadata.get("content_type", "text"),
                        "token_count": len(doc.page_content.split())
                    },
                    "_id": doc.metadata.get("chunk_id", idx)
                }
                results.append(result)
        
        return results
    except Exception as e:
        print(f"Keyword search error: {e}")
        return []


def get_query_embedding(query_text, embedding_model="nomic-embed-text"):
    """Get embedding for query using nomic-embed-text model"""
    try:
        embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url="http://localhost:11434"
        )
        query_embedding = embeddings.embed_query(query_text)
        return query_embedding
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return None


def semantic_search(query_text, top_k=20):
    """
    Perform semantic search using vector embeddings with nomic-embed-text.
    Args:
        query_text (str): The query text to search for
        top_k (int): Number of results to return
    Returns:
        list: Search results in OpenSearch-like format
    """
    try:
        vector_store = load_faiss_index("faiss_index")

        results = vector_store.similarity_search_with_score(query_text, k=top_k)
        
        formatted_results = []
        for doc, score in results:
            similarity_score = 1.0 / (1.0 + score)
            
            result = {
                "_score": float(similarity_score),
                "_source": {
                    "content": doc.page_content,
                    "content_type": doc.metadata.get("content_type", "text"),
                    "token_count": len(doc.page_content.split())
                },
                "_id": doc.metadata.get("chunk_id", 0)
            }
            formatted_results.append(result)
        
        return formatted_results
    except Exception as e:
        print(f"Semantic search error: {e}")
        return []


def hybrid_search(query_text, top_k=20):
    """
    Perform hybrid search using both keyword and semantic search.
    Args:
        query_text (str): The query text to search for
        top_k (int): Number of results to return
    Returns:
        list: Search results in OpenSearch-like format
    """
    try:
        semantic_results = semantic_search(query_text, top_k)
        keyword_results = keyword_search(query_text, top_k)

        combined_results = {}

        for result in semantic_results:
            doc_id = result["_id"]
            combined_results[doc_id] = result.copy()
            combined_results[doc_id]["semantic_score"] = result["_score"]
            combined_results[doc_id]["keyword_score"] = 0.0

        for result in keyword_results:
            doc_id = result["_id"]
            if doc_id in combined_results:
                combined_results[doc_id]["keyword_score"] = result["_score"]
            else:
                combined_results[doc_id] = result.copy()
                combined_results[doc_id]["semantic_score"] = 0.0
                combined_results[doc_id]["keyword_score"] = result["_score"]

        final_results = []
        
        for doc_id, result in combined_results.items():
            semantic_score = result.get("semantic_score", 0)
            keyword_score = result.get("keyword_score", 0)

            hybrid_score = (0.7 * semantic_score) + (0.3 * keyword_score)
            
            final_result = {
                "_score": float(hybrid_score),
                "_source": result["_source"],
                "_id": doc_id
            }
            final_results.append(final_result)

        final_results.sort(key=lambda x: x["_score"], reverse=True)
        return final_results[:top_k]
        
    except Exception as e:
        print(f"Hybrid search error: {e}")
        try:
            return keyword_search(query_text, top_k)
        except Exception as e2:
            print(f"Fallback search error: {e2}")
            return []


if __name__ == "__main__":
    from pprint import pprint
    
    query = "What datasets were used to evaluate the Paper."
    
    results = keyword_search(query, top_k=5)
    results = semantic_search(query, top_k=5)
    results = hybrid_search(query, top_k=5)
    #pprint(results)