# Email-LLM

Natural language search for your Thunderbird emails using local LLMs.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Natural Language Search**: Ask questions like "Find invoices from last month" or "What did John say about the project?"
- **Local LLM Integration**: Works with LM Studio or Ollama - your data stays on your machine
- **Full Attachment Support**: Extracts and searches text from PDFs, Word docs, Excel files, and PowerPoint
- **Image Analysis**:
  - **YOLO**: Fast object detection to filter images (find photos with people, cars, etc.)
  - **Vision Models**: Deep image understanding with LLaVA for detailed descriptions
  - **OCR**: Extract text from images and scanned documents
- **RAG-Powered Answers**: Get AI-generated answers based on your actual emails
- **Web UI & CLI**: Use whichever interface you prefer

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Thunderbird    │────▶│  Email Indexer   │────▶│  Vector DB      │
│  (mbox/maildir) │     │  (Python)        │     │  (ChromaDB)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                        │
                        ┌───────┴───────┐                │
                        ▼               ▼                │
                 ┌──────────┐   ┌──────────────┐         │
                 │ Text     │   │ Attachments  │         │
                 │ Embedding│   │ (PDF/OCR/    │         │
                 │          │   │  Vision)     │         │
                 └──────────┘   └──────────────┘         │
                                                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Web UI / CLI   │◀───▶│  LM Studio       │◀───▶│  RAG Query      │
│                 │     │  (LLM + Vision)  │     │  Engine         │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Thunderbird email client
- One of:
  - [LM Studio](https://lmstudio.ai/) (recommended)
  - [Ollama](https://ollama.ai/)
- Optional: Tesseract OCR for image text extraction

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/email-llm.git
cd email-llm

# Run the setup script
./setup.sh

# Activate the virtual environment
source venv/bin/activate

# Check your configuration
email-llm check
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the package
pip install -e .

# Copy and edit configuration
cp .env.example .env
# Edit .env with your settings
```

### System Dependencies

```bash
# Ubuntu/Debian - Install Tesseract for OCR
sudo apt install tesseract-ocr

# For Ollama vision models
ollama pull llava:13b
ollama pull nomic-embed-text
ollama pull mistral
```

## Configuration

Edit `.env` or set environment variables:

```bash
# Thunderbird profile path (Snap installation)
EMAIL_LLM_THUNDERBIRD_PROFILE_PATH=~/snap/thunderbird/common/.thunderbird

# LLM backend: 'lmstudio' or 'ollama'
EMAIL_LLM_LLM_BACKEND=lmstudio

# LM Studio (default)
EMAIL_LLM_LMSTUDIO_BASE_URL=http://localhost:1234/v1

# Ollama (alternative)
EMAIL_LLM_OLLAMA_BASE_URL=http://localhost:11434
EMAIL_LLM_OLLAMA_MODEL=mistral
EMAIL_LLM_OLLAMA_VISION_MODEL=llava:13b
```

### Finding Your Thunderbird Profile

```bash
# Standard installation
~/.thunderbird/

# Snap installation
~/snap/thunderbird/common/.thunderbird/

# Flatpak installation
~/.var/app/org.mozilla.Thunderbird/.thunderbird/
```

## Usage

### Index Your Emails

First, index your emails to make them searchable:

```bash
# Index all emails
email-llm index

# Re-index from scratch
email-llm index --reindex

# Index with deep image analysis (slower but more accurate)
email-llm index --vision

# Index only first 100 emails (for testing)
email-llm index --max 100
```

### Search via CLI

```bash
# Natural language search
email-llm search "find invoices from Amazon"

# Ask a question and get an AI answer
email-llm ask "What was the total on the invoice from Acme Corp?"

# Search without AI answer generation
email-llm search "project meeting notes" --no-llm

# Use Ollama instead of LM Studio
email-llm search "vacation requests" --backend ollama
```

### Web Interface

```bash
# Start the web server
email-llm serve

# Custom host/port
email-llm serve --host 0.0.0.0 --port 8080
```

Then open http://localhost:8000 in your browser.

### Other Commands

```bash
# Check system configuration
email-llm check

# View index statistics
email-llm stats

# List objects YOLO can detect
email-llm objects
```

## Search Filters

The web UI and API support advanced filters:

- **From Address**: Filter by sender
- **Date Range**: Search within specific dates
- **Folder**: Search specific mailbox folders
- **Attachment Type**: Filter by image, PDF, Word, Excel, etc.
- **Has Attachments**: Only show emails with attachments
- **Contains Object**: Filter images by detected objects (person, car, dog, etc.)
- **Has Faces**: Filter images containing people

## API

The web server exposes a REST API:

```bash
# Search emails
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "invoices", "limit": 10}'

# Ask a question
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What invoices did I receive last month?"}'

# Get statistics
curl http://localhost:8000/api/stats

# Health check
curl http://localhost:8000/api/health
```

## How It Works

1. **Parsing**: Reads Thunderbird's mbox files and extracts emails with metadata
2. **Attachment Extraction**: Extracts text from PDFs, Office documents, and images
3. **Image Analysis**:
   - YOLO detects objects for fast filtering
   - OCR extracts visible text
   - Vision models provide detailed descriptions
4. **Embedding**: Converts text into vector embeddings using sentence-transformers
5. **Storage**: Stores embeddings in ChromaDB for fast similarity search
6. **Search**: Finds relevant emails using semantic similarity
7. **RAG**: Sends relevant context to the LLM to generate natural language answers

## Models Used

| Purpose | Default Model | Alternative |
|---------|--------------|-------------|
| Embeddings | all-MiniLM-L6-v2 | nomic-embed-text (Ollama) |
| LLM | Any LM Studio model | mistral (Ollama) |
| Vision | llava:13b | bakllava |
| Object Detection | YOLOv8n | YOLOv8s/m/l/x |

## Privacy

- **100% Local**: All processing happens on your machine
- **No Cloud**: No data is sent to external servers
- **Your Data**: Embeddings are stored locally in ChromaDB

## Troubleshooting

### "Thunderbird profile not found"

Check your profile path:
```bash
ls ~/snap/thunderbird/common/.thunderbird/
ls ~/.thunderbird/
```

### "LM Studio not connected"

1. Make sure LM Studio is running
2. Load a model in LM Studio
3. Check the server is enabled (Settings → Local Server)

### "Ollama not connected"

```bash
# Start Ollama
ollama serve

# Pull required models
ollama pull mistral
ollama pull llava:13b
```

### OCR not working

```bash
# Install Tesseract
sudo apt install tesseract-ocr

# Verify installation
tesseract --version
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
ruff check src/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
