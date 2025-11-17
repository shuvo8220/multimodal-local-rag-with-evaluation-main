A comprehensive **Retrieval-Augmented Generation (RAG)** system that processes PDF documents containing text, images, and tables. Users can upload PDFs and ask natural language questions about their content through an intuitive web interface.

## 🌟 Features

### 📄 **Multimodal Document Processing**

- **Text Extraction**: Semantic chunking with title-based organization
- **Image Analysis**: AI-powered visual understanding using GPT-4o-mini
- **Table Processing**: Intelligent table parsing and summarization using DeepSeek
- **High-Resolution Processing**: Advanced document layout analysis

### 🔍 **Advanced Search Capabilities**

- **Semantic Search**: Vector-based similarity using embedding models
- **Keyword Search**: Traditional BM25-based lexical matching
- **Hybrid Search**: Combined approach for optimal results (70% semantic + 30% keyword)

### 🧠 **AI-Powered Generation**

- **Local LLM Integration**: Uses DeepSeek model via Ollama
- **Context-Aware Responses**: Structured prompts with document citations
- **Multimodal Understanding**: Combines text, visual, and tabular information

### 🎨 **User-Friendly Interface**

- **Drag-and-Drop Upload**: Simple PDF file handling
- **Real-Time Processing**: Live status updates during document ingestion
- **Interactive Querying**: Multiple search methods with instant responses
- **Clean Design**: Professional Gradio-based web interface

## 🏗️ Architecture

```
📂 Project Structure
├── app.py                    # Main Gradio web interface
├── get_chunks.py            # Document processing and content extraction
├── ingesion_pipeline.py     # Vector database creation and storage
├── retrieval_pipeline.py    # Search methods (semantic, keyword, hybrid)
├── generator_pipeline.py    # Response generation with LLM
├── helper.py               # Utility functions (embeddings, etc.)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env                   # Environment variables
└── faiss_index/          # Vector database storage (auto-created)
```

## 📋 Prerequisites

### System Requirements

- **Python 3.8+**
- **8GB+ RAM** (recommended for processing large PDFs)
- **5GB+ free disk space** (for models and vector storage)

### External Dependencies

- **Ollama** (for local LLM and embeddings)
- **OpenAI API Key** (for image processing with GPT-4o-mini)
- **Tesseract OCR** (for text extraction from images)

## 🚀 Installation Guide

### 1. **Clone the Repository**

```bash
git clone <your-repository-url>
cd multimodal-rag-system
```

### 2. **Set Up Python Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. **Install Python Dependencies**

```bash
pip install -r requirements.txt
```

### 4. **Install Ollama**

#### Option A: Standard Installation

**macOS:**

```bash
brew install ollama
```

**Linux:**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai) and run the installer.

#### Option B: Docker Installation

```bash
# Pull Ollama Docker image
docker pull ollama/ollama

# Run Ollama container
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  ollama/ollama

# Access container to pull models
docker exec -it ollama ollama pull deepseek-r1:1.5b
docker exec -it ollama ollama pull nomic-embed-text
```

### 5. **Install Required Models**

```bash
# Start Ollama service
ollama serve

# In a new terminal, install models
ollama pull deepseek-r1:1.5b      # Main LLM for response generation
ollama pull nomic-embed-text       # Embedding model for semantic search
```

### 6. **Install Tesseract OCR**

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**

```bash
brew install tesseract
```

**Windows:**
Download from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)

### 7. **Set Up Environment Variables**

Create a `.env` file in the project root:

```bash
# OpenAI API Key (required for image processing)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434
```

## 🔧 Configuration

### Model Configuration

The system uses these models by default:

- **LLM**: `deepseek-r1:1.5b` (response generation)
- **Embeddings**: `nomic-embed-text` (semantic search)
- **Vision**: `gpt-4o-mini` (image analysis)

### Processing Parameters

```python
# Text chunking settings
max_characters=2000          # Maximum chunk size
min_chars_to_combine=500     # Minimum size to combine chunks
chars_before_new_chunk=1500  # Overlap between chunks

# Search settings
top_k=5                      # Number of chunks to retrieve
hybrid_weights=(0.7, 0.3)    # Semantic vs keyword weight
```

## 🏃‍♂️ How to Run

### 1. **Start Required Services**

```bash
# Terminal 1: Start Ollama (if not using Docker)
ollama serve

# Terminal 2: Verify models are available
ollama list
```

### 2. **Launch the Application**

```bash
python app.py
```

### 3. **Access the Web Interface**

- Open your browser to: **http://localhost:7860**
- The interface should automatically open in your default browser

### 4. **Using the System**

1. **Upload PDF**: Drag and drop a PDF file into the upload area
2. **Process Document**: Click "🔄 Process PDF" and wait for completion
3. **Select Search Method**: Choose from semantic, keyword, or hybrid
4. **Ask Questions**: Type your question and click "🚀 Get Answer"
5. **View Results**: The AI-generated response will appear in the response area

## 💡 Usage Examples

### Sample Questions

```
- "What is the main topic of this document?"
- "Summarize the key findings from the tables"
- "What do the images show?"
- "What datasets were used in this research?"
- "What are the main conclusions?"
- "Explain the methodology described in the paper"
```

### Search Method Comparison

- **Semantic Search**: Best for conceptual queries and finding related content
- **Keyword Search**: Best for finding specific terms and exact matches
- **Hybrid Search**: Balanced approach, recommended for most queries

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'unstructured'`

```bash
pip install unstructured[pdf]
```

**Issue**: `ConnectionError: Could not connect to Ollama`

```bash
# Make sure Ollama is running
ollama serve
# Or check if using Docker
docker ps | grep ollama
```

**Issue**: `OPENAI_API_KEY not found`

```bash
# Create .env file with your API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

**Issue**: `Error: No embedding model found`

```bash
ollama pull nomic-embed-text
```

**Issue**: PDF processing fails

```bash
# Install additional dependencies
pip install pdf2image poppler-utils
```

**Issue**: Button not clickable after PDF processing

- Check browser console (F12) for debug messages
- Ensure Ollama models are properly loaded
- Verify .env file contains valid OpenAI API key

### Performance Optimization

**For Large PDFs (>50 pages):**

- Reduce `max_characters` to 1000
- Increase `top_k` to 7-10 for better context
- Use semantic search for better accuracy

**For Memory Issues:**

- Use `faiss-cpu` instead of `faiss-gpu`
- Process smaller chunks with `max_characters=1000`
- Clear sessions frequently using "🗑️ Clear Session"

**For Speed Improvements:**

- Use Docker for Ollama (better resource management)
- Consider upgrading to larger embedding models
- Cache processed documents for repeated use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed descriptions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the debug messages in browser console
3. Ensure all dependencies are properly installed
4. Create an issue on the GitHub repository

## 🙏 Acknowledgments

- **Unstructured.io** for document processing capabilities
- **Ollama** for local LLM infrastructure
- **LangChain** for RAG framework components
- **Gradio** for the user interface framework
- **OpenAI** for advanced image understanding
- **FAISS** for efficient vector search

---
