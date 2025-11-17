import gradio as gr
import os
import tempfile
import shutil
from pathlib import Path
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from get_chunks import process_images_with_captions, process_table_with_captions, process_raw_text
from ingesion_pipeline import ingest_all_content_into_faiss
from generator_pipeline import generate_rag_response


class MultimodalRAGApp:
    
    def __init__(self):
        """Initialize app with default states"""
        self.current_pdf_path = None
        self.is_pdf_processed = False
        self.vector_store_ready = False
        
    def process_uploaded_pdf(self, pdf_file):

        if pdf_file is None:
            return "❌ No PDF file uploaded.", "⚠️ Please upload a PDF file first.", False, False
        
        try:

            temp_dir = tempfile.mkdtemp()
            pdf_path = os.path.join(temp_dir, "uploaded_document.pdf")
            shutil.copy2(pdf_file.name, pdf_path)
            self.current_pdf_path = pdf_path
            
            status_message = "📄 PDF uploaded successfully!\n"

            status_message += "🔍 Extracting content from PDF...\n"
            raw_chunks = partition(
                filename=pdf_path,
                strategy="hi_res",  
                infer_table_structure=True,
                extract_image_block_types=["Figure", "Table", "Image"],
                extract_image_block_to_payload=True,
                chunking_strategy=None,
            )
            
            status_message += "🖼️ Processing images...\n"
            processed_images = process_images_with_captions(raw_chunks, llm=True)
            status_message += f"   ✅ Processed {len(processed_images)} images\n"
            
            status_message += "📊 Processing tables...\n"
            processed_tables, table_errors = process_table_with_captions(raw_chunks, ollama=True)
            status_message += f"   ✅ Processed {len(processed_tables)} tables\n"
            if table_errors:
                status_message += f"   ⚠️ {len(table_errors)} table processing errors\n"
            

            status_message += "📝 Processing text content...\n"
            chunks = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",
                chunking_strategy="by_title",  
                max_characters=2000,          
                min_chars_to_combine=500,      
                chars_before_new_chunk=1500,  
            )
            semantic_chunks = process_raw_text(chunks, llm=False)
            status_message += f"   ✅ Created {len(semantic_chunks)} text chunks\n"
            

            status_message += "🗄️ Creating vector database...\n"
            vector_store = ingest_all_content_into_faiss(
                processed_images, 
                processed_tables, 
                semantic_chunks,
                save_path="faiss_index"  
            )
            
            total_chunks = len(processed_images + processed_tables + semantic_chunks)
            status_message += "✅ Vector database created successfully!\n"
            status_message += f"📊 Total content ingested: {total_chunks} chunks\n"
            status_message += "\n🚀 Ready for querying!"
            
            # Update app state
            self.is_pdf_processed = True
            self.vector_store_ready = True
            
            return status_message, "✅ PDF processed successfully! You can now ask questions.", True, True
            
        except Exception as e:
            error_msg = f"❌ Error processing PDF: {str(e)}"
            return error_msg, error_msg, False, False
    
    def handle_query(self, query, search_type):



        if not query.strip():
            return "⚠️ Please enter a question."
        
        if not self.vector_store_ready:
            return "⚠️ Please upload and process a PDF file first."
        
        try:
            response = generate_rag_response(
                query=query,
                search_type=search_type,
                top_k=5,                   
                model_name="gpt-4o-mini", 
             
            )
            
            return response
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def clear_session(self):

        try:
            if self.current_pdf_path and os.path.exists(self.current_pdf_path):
                temp_dir = os.path.dirname(self.current_pdf_path)
                shutil.rmtree(temp_dir, ignore_errors=True)
            

            self.current_pdf_path = None
            self.is_pdf_processed = False
            self.vector_store_ready = False
            
            return (
                "🔄 Session cleared successfully!", 
                "",      
                "",      
                False,   
                False   
            )
            
        except Exception as e:
            return f"❌ Error clearing session: {str(e)}", "", "", False, False


def create_interface():

    # Initialize the main application
    app = MultimodalRAGApp()
    
    # Create Gradio interface with custom styling
    with gr.Blocks(
        title="🤖 Multimodal RAG System",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        .main {
            max-width: 1200px !important;
            margin: 0 auto !important;
            padding: 20px !important;
        }
        .center-content {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        """
    ) as demo:
        
        with gr.Column(elem_classes=["center-content"]):
            
            gr.Markdown("# 🤖 Multimodal RAG System")
            gr.Markdown("Upload a PDF document and ask questions about its content, including text, images, and tables!")
            
            # Main interface layout
            with gr.Row():
                
                # Left column: PDF upload and processing
                with gr.Column(scale=1):
                    gr.Markdown("## 📄 Step 1: Upload PDF Document")
                    
                    
                    pdf_input = gr.File(
                        label="Drag and drop your PDF file here",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    
                    
                    with gr.Row():
                        process_btn = gr.Button("🔄 Process PDF", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ Clear Session", variant="secondary")
                    
                    # Processing status display
                    processing_status = gr.Textbox(
                        label="Processing Status",
                        placeholder="Upload a PDF to begin processing...",
                        lines=10,
                        max_lines=15,
                        interactive=False
                    )
                
                # Right column: Query interface
                with gr.Column(scale=1):
                    gr.Markdown("## 💬 Step 2: Ask Questions")
                    
                    # Search method selection
                    search_type = gr.Radio(
                        choices=["semantic", "keyword", "hybrid"],
                        label="Search Method",
                        value="hybrid",  # Default to hybrid search
                        info="Choose how to search through your document"
                    )
                    
                    # Query input
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about the uploaded document...",
                        lines=2,
                        interactive=True
                    )
                    
                    # Query submission button (starts disabled)
                    query_btn = gr.Button(
                        "🚀 Get Answer", 
                        variant="secondary",  
                        size="lg",
                        interactive=False
                    )
                    
                    # Response display
                    response_output = gr.Textbox(
                        label="Response",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        placeholder="Answers will appear here after you upload a PDF and ask a question..."
                    )
            
            pdf_processed = gr.State(False)
            query_ready = gr.State(False)
            
            # Expandable help sections
            with gr.Accordion("💡 Example Questions", open=False):
                gr.Markdown("""
                Try these example questions after uploading your PDF:
                - "What functional requirements are listed in the SRS, and how do they collectively support the management of trainees, trainers, courses, enrollment, user roles, and reporting?"
                - "Summarize the key findings from the tables"
                - "What do the images show?"
                - "What datasets were used in this research?"
                - "What are the main conclusions?"
                """)
            
            with gr.Accordion("ℹ️ Search Methods Explained", open=False):
                gr.Markdown("""
                - **Semantic Search**: Uses AI embeddings to find contextually similar content
                - **Keyword Search**: Traditional text matching using BM25 algorithm  
                - **Hybrid Search**: Combines both methods for best results (recommended)
                """)
        
        # Event handler functions
        def update_ui_after_processing(pdf_file):
            """Handle PDF processing and update UI components"""
            print(f"Processing PDF: {pdf_file}") 
            
            # Process the PDF through the complete pipeline
            status, short_status, processed, ready = app.process_uploaded_pdf(pdf_file)
            
            print(f"Processing results - processed: {processed}, ready: {ready}")  # Debug
            
            # Update button state based on processing results
            button_update = gr.update(
                interactive=ready,
                variant="primary" if ready else "secondary"
            )
            
            print(f"Button update: {button_update}")  # Debug
            
            return [
                status,      
                button_update, 
                processed,    
                ready        
            ]
        
        def handle_query_with_validation(query, search_type, pdf_processed_state, query_ready_state):
            """Handle user queries with proper validation"""
            print(f"Query handler - PDF processed: {pdf_processed_state}, Query ready: {query_ready_state}")
            print(f"App states - is_pdf_processed: {app.is_pdf_processed}, vector_store_ready: {app.vector_store_ready}")
            

            if not app.is_pdf_processed or not app.vector_store_ready:
                return "⚠️ Please upload and process a PDF file first."
            
            if not query.strip():
                return "⚠️ Please enter a question."
            

            return app.handle_query(query, search_type)
        
        def clear_all():
            """Clear session and reset all UI components"""
            status_msg, query_clear, response_clear, pdf_state, ready_state = app.clear_session()
            
            # Reset button to disabled state
            button_reset = gr.update(interactive=False, variant="secondary")
            
            return [
                status_msg,    
                query_clear,  
                response_clear,
                button_reset,  
                pdf_state,     
                ready_state   
            ]
        

        process_btn.click(
            fn=update_ui_after_processing,
            inputs=[pdf_input],
            outputs=[processing_status, query_btn, pdf_processed, query_ready]
        )
        
        # Query submission events
        query_btn.click(
            fn=handle_query_with_validation,
            inputs=[query_input, search_type, pdf_processed, query_ready],
            outputs=[response_output]
        )
        
        # Allow Enter key to submit query
        query_input.submit(
            fn=handle_query_with_validation,
            inputs=[query_input, search_type, pdf_processed, query_ready],
            outputs=[response_output]
        )
        
        # Clear session event
        clear_btn.click(
            fn=clear_all,
            outputs=[processing_status, query_input, response_output, query_btn, pdf_processed, query_ready]
        )
    
    return demo


if __name__ == "__main__":
    """
    Launch the application
    Run this script and navigate to http://localhost:7860 in your browser
    """
    print("🚀 Starting Multimodal RAG System...")
    print("📋 Make sure these services are running:")
    print("   - Ollama server (ollama serve)")
    print("   - DeepSeek model (ollama pull deepseek-r1:1.5b)")
    print("   - Embedding model (ollama pull nomic-embed-text)")
    print("   - OpenAI API key in .env file (for image processing)")
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",  # Local access
        server_port=7860,         # Default Gradio port
        share=False,              # Set to True for public sharing
        debug=True,               # Enable debug mode
        inbrowser=True            # Auto-open browser
    )