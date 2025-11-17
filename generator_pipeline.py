import json
import re
import os
import openai
from langchain.prompts import PromptTemplate
from retrieval_pipeline import hybrid_search, keyword_search, semantic_search

# Make sure your OpenAI API key is set
# export OPENAI_API_KEY="your_key_here"  # Linux/macOS
# setx OPENAI_API_KEY "your_key_here"    # Windows

openai.api_key = os.getenv("OPENAI_API_KEY")

# === RAG GPT Prompt Template ===
RAG_PROMPT_TEMPLATE = """Answer the question using the documents below. Be specific and include relevant details.

If the answer is not in the documents, say "I cannot answer based on the provided documents."

Format the answer as JSON with keys: "answer", "supporting_documents"

Documents:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)

def clean_gpt_response(response_text):
    """Clean GPT responses if needed"""
    cleaned = re.sub(r'\n\s*\n', '\n\n', response_text)
    return cleaned.strip()

def generate_with_gpt(prompt_text, model_name="gpt-4o-mini", temperature=0.2, max_tokens=500):
    """Generate response using OpenAI GPT (new API v1)"""
    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        raw_response = response.choices[0].message.content
        return clean_gpt_response(raw_response)
    except Exception as e:
        return f"Error generating response from GPT: {str(e)}"

def format_retrieved_context(results):
    """Format retrieved documents into structured context"""
    if not results:
        return "No documents retrieved."
    
    contexts = []
    for i, hit in enumerate(results, 1):
        source = hit["_source"]
        content = source.get("content", "").strip()
        if content:
            doc_entry = f"Document {i}:\n{content}"
            contexts.append(doc_entry)
    return "\n\n---\n\n".join(contexts)

def generate_rag_response(query, search_type="semantic", top_k=5, model_name="gpt-4o-mini"):
    """Generate RAG response using GPT"""
    try:
        # Retrieve documents
        if search_type == "keyword":
            results = keyword_search(query, top_k=top_k)
        elif search_type == "semantic":
            results = semantic_search(query, top_k=top_k)
        else:
            results = hybrid_search(query, top_k=top_k)

        if not results:
            return "No relevant documents found."

        # Format context and build prompt
        context_text = format_retrieved_context(results)
        prompt_text = prompt.format(context=context_text, question=query)

        # Generate GPT response
        response = generate_with_gpt(prompt_text, model_name=model_name)
        return response

    except Exception as e:
        return f"Error in RAG process: {str(e)}"

def interactive_rag():
    """Interactive RAG session"""
    print("=== Interactive RAG System (GPT) ===")
    print("Available search types: semantic, keyword, hybrid")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("\nEnter your question: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
                
        search_type = input("Search type (default: hybrid): ").strip() or "hybrid"
        top_k = input("Number of documents (default: 5): ").strip() or "5"
        
        try:
            top_k = int(top_k)
        except ValueError:
            top_k = 5
            
        response = generate_rag_response(query=query, search_type=search_type, top_k=top_k)
        print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    # Example test queries
    test_queries = [
        "What datasets were used to evaluate the paper?",
        "What are the main contributions of the study?"
    ]
    
    print("=== RAG System Test (GPT) ===")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest Query {i}: {query}")
        response = generate_rag_response(query=query, search_type="hybrid", top_k=5)
        print(f"Response:\n{response}\n")
