import base64
import os
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element, Text, FigureCaption, Image, Table, Title, NarrativeText, ListItem,  CompositeElement
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")



def process_images_with_captions(raw_chunks, llm=True):
    processed_images = []  # Fixed variable name
    
    for idx, chunk in enumerate(raw_chunks):
        if isinstance(chunk, Image):
            if idx + 1 < len(raw_chunks) and isinstance(raw_chunks[idx + 1], FigureCaption):
                caption = raw_chunks[idx + 1].text
            else:
                caption = None
            
            # Handle missing metadata attributes safely
            try:
                base64_image = chunk.metadata.image_base64 if hasattr(chunk.metadata, 'image_base64') else None
                page_number = chunk.metadata.page_number if hasattr(chunk.metadata, 'page_number') else None
                file_name = chunk.metadata.file_name if hasattr(chunk.metadata, 'file_name') else None
            except AttributeError:
                base64_image = None
                page_number = None
                file_name = None
                
            image_info = {
                "caption": caption if caption else "No caption",
                "image_text": chunk.text if hasattr(chunk, 'text') else "",
                "base64_image": base64_image,
                "content": chunk.text if hasattr(chunk, 'text') else "",
                "page_number": page_number,
                "file_name": file_name
            }
            
            if llm and base64_image:  # Only process if we have base64 image
                try:
                    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    
                    messages = [
                        HumanMessage(content=[
                            {
                                "type": "text",
                                "text": (
                                    "You are a multimodal AI specialized in reading business and scientific documents. "
                                    "Carefully analyze the provided image, which may be a graph, chart, table, scanned figure, or illustration. "
                                    "\n\n"
                                    "Instructions:\n"
                                    "1. First, identify the type of image (e.g., line graph, bar chart, scatter plot, table, scanned text, diagram, illustration).\n"
                                    "2. If it is a graph or chart: extract axes labels, legends, units, and describe the overall trends, patterns, or anomalies.\n"
                                    "3. If it contains text (OCR), transcribe the text accurately.\n"
                                    "4. If it is an illustration/diagram, describe the main elements and their relationships.\n"
                                    "5. Provide a detailed structured summary of the key insights in plain language.\n\n"
                                    f"Additional context:\n"
                                    f"- Caption: {image_info['caption']}\n"
                                    f"- Extracted text (if available): {image_info['image_text']}\n\n"
                                    "Return only the analysis without extra commentary."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_info['base64_image']}"
                                }
                            }
                        ])
                    ]
                    response = model.invoke(messages)
                    image_info["content"] = response.content
                except Exception as e:
                    print(f"Error processing image with LLM: {e}")
                    
                    
            processed_images.append(image_info)  
            
    return processed_images


def process_table_with_captions(raw_chunks, ollama=True):
    processed_tables = []
    encountered_errors = []
    
    for idx, chunk in enumerate(raw_chunks):
        if isinstance(chunk, Table):
            # Handle missing metadata attributes safely
            try:
                text_as_html = chunk.metadata.text_as_html if hasattr(chunk.metadata, 'text_as_html') else None
                filename = chunk.metadata.filename if hasattr(chunk.metadata, 'filename') else ""
            except AttributeError:
                text_as_html = None
                filename = ""
            
            # Store table data
            table_data = {
                "table_as_html": text_as_html,
                "table_text": chunk.text if hasattr(chunk, 'text') else "",
                "content": chunk.text if hasattr(chunk, 'text') else "",
                "content_type": "table",
                "filename": filename,
                "page_number": chunk.metadata.page_number if hasattr(chunk.metadata, 'page_number') else None
                
            }
            
            if ollama and text_as_html:
                try:
                    import requests
                    url = "http://localhost:11434/api/generate"
                    data = {
                        "model": "deepseek-r1:1.5b",
                        "prompt": (
                            "You are an expert data analyst. Analyze the HTML table below and provide ONLY a clear, structured summary.\n\n"
                            "Instructions:\n"
                            "1. Identify the table structure (columns, rows, headers)\n"
                            "2. Extract key data points and numerical values\n"
                            "3. Identify patterns, trends, or notable findings\n"
                            "4. Summarize the main insights in plain language\n"
                            "5. Do NOT include any reasoning steps, thinking process, or <think></think> tags\n"
                            "6. Provide ONLY the final summary in a clear, professional format\n\n"
                            f"HTML Table:\n{text_as_html}\n\n"
                            "Summary:"
                        ),
                        "max_tokens": 1000,
                        "stream": False,
                        "temperature": 0.1,
                    }

                    response = requests.post(url, json=data)
                    response.raise_for_status()

                    raw_response = response.json().get("response", "No response from model")
                    
                    # Clean the response to remove thinking tags and extract only summary
                    if "<think>" in raw_response and "</think>" in raw_response:
                        # Extract content after </think> tag
                        summary = raw_response.split("</think>")[-1].strip()
                    else:
                        summary = raw_response.strip()
                    
                    # Remove any remaining thinking patterns
                    summary = summary.replace("<think>", "").replace("</think>", "").strip()
                    
                    table_data["content"] = summary
                except Exception as e:
                    encountered_errors.append(
                        {
                            "error": str(e),
                            "error_message": "Error generating description with Ollama.",
                        }
                    )

            processed_tables.append(table_data)

    print(f"Processed {len(processed_tables)} tables with descriptions")
    print(f"Errors encountered: {len(encountered_errors)}")
    return processed_tables, encountered_errors

            


def process_raw_text(text_chunks, llm=False):
    # Convert to more usable format
    processed_chunks = []

    for idx, chunk in enumerate(text_chunks):
        if isinstance(chunk, CompositeElement):
            try:
                filename = chunk.metadata.filename if hasattr(chunk.metadata, 'filename') else ""
            except AttributeError:
                filename = ""
            
            chunk_data = {
                "content": chunk.text if hasattr(chunk, 'text') else "",
                "content_type": "text",
                "filename": filename,
            }
            processed_chunks.append(chunk_data)

    print(f"Created {len(processed_chunks)} semantic chunks from document")
    return processed_chunks




if __name__ == "__main__":

    pdf_path = "files/Attention Is All You Need.pdf"

    try:
        # --- First: Partition PDF into raw chunks (images, tables, text) ---
        raw_chunks = partition(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Figure", "Table", "Image"],
            extract_image_block_to_payload=True,
            chunking_strategy=None,
        )

        processed_images = process_images_with_captions(raw_chunks, llm=True)
        print(f"Processed {len(processed_images)} images")

        processed_tables, table_errors = process_table_with_captions(raw_chunks, ollama=True)
        print(f"Processed {len(processed_tables)} tables, Errors: {len(table_errors)}")
        
        chunks = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            chunking_strategy="by_title",
            max_characters=2000,
            min_chars_to_combine=500,
            chars_before_new_chunk=1500,
        )
        semantic_chunks = process_raw_text(chunks, llm=False)
        
        
        for idx, chunk in enumerate(semantic_chunks):
            if isinstance(chunk, CompositeElement):
                print(f"Chunk {idx}: {chunk.text[:50]}...")

    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
    except Exception as e:
        print(f"Error processing PDF: {e}")

    