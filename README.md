# Retrobot 

A retro-themed AI chatbot web application powered by Ollama with local LLM support. Features enhanced chat modes, image processing, persistent chat history, and user customization.

##  Features

- ** Retro 8-bit UI Design** - Nostalgic gaming aesthetic with pixelated styling
- ** Ollama Integration** - Run AI locally using any Ollama-supported model
- ** Smart Chat Modes** - Auto-detects query type: coding, research, tutorial, writing, creative, debugging
- ** Image Processing** - Upload images, extract text with OCR, analyze with AI
- ** Persistent Chat History** - All conversations saved locally as JSON
- ** Session Management** - Tags, favorites, search, rename, and export chats
- ** Export Options** - Export conversations as Markdown, PDF, or JSON
- ** User Customization** - Theme selection, font sizing, difficulty levels, settings
- ** Code Syntax Highlighting** - Formatted code display with language detection
- ** Intelligent Prompts** - Context-aware system prompts for each chat mode

##  Quick Start

### Prerequisites

- Python 3.8+
- Ollama installed and running (`ollama serve`)
- A model downloaded: `ollama pull deepseek-llm` (or any supported model)

### Installation

```bash
# Clone repository
git clone https://github.com/RHENIU5/Retrobot.git
cd Retrobot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `.env` file with your Ollama settings:

```env
# Ollama Configuration
ollama_endpoint=http://localhost:11434

# Model Configuration  
# Available models: deepseek-llm, neural-chat, llama3, mistral, openchat
# Install with: ollama pull <model-name>
OLLAMA_MODEL=deepseek-llm
```

### Running

```bash
python app.py
```

Then open your browser to: `http://127.0.0.1:8000`

## Dependencies

- **Flask 3.0.0** - Web framework
- **OpenAI SDK 1.51.0+** - Ollama API compatibility
- **Python-dotenv 1.0.0** - Environment variables
- **Pillow 10.0.0+** - Image processing
- **pytesseract 0.3.10+** - OCR text extraction
- **ReportLab 4.0.0** - PDF generation
- **Markdown 3.5.0** - Markdown parsing
- **Pygments 2.15.0** - Syntax highlighting

## Chat Modes

| Mode | Detection | System Prompt | Use Case |
|------|-----------|---------------|----------|
|  **Coding** | Code keywords, syntax | Expert code generator | Programming tasks |
|  **Research** | Research keywords | Expert researcher | Information gathering |
|  **Tutorial** | Learning keywords | Expert educator | Step-by-step guides |
|  **Writing** | Writing keywords | Expert writing assistant | Content creation |
|  **Creative** | Brainstorm keywords | Creative ideation | Brainstorming |
|  **Debugging** | Debug keywords | Debugging expert | Issue resolution |
|  **Chat** | Default | Helpful assistant | General conversation |

##  Project Structure

```
Retrobot/
├── app.py                 # Flask backend & Ollama integration
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
├── .gitignore            # Git ignore rules
├── chat_history.json     # Persistent chat storage
├── user_settings.json    # User preferences
│
├── templates/
│   └── index.html        # Frontend UI (Retro theme)
│
└── static/
    └── style.css         # Retro 8-bit styling
```

##  API Endpoints

### Chat
- `POST /chat` - Send message and get AI response

### Sessions
- `GET /sessions` - List all chat sessions
- `GET /sessions/<id>` - Get session details
- `PUT /sessions/<id>` - Update session name
- `DELETE /sessions/<id>` - Delete session
- `POST /clear` - Start new session

### Features
- `POST /sessions/<id>/favorite` - Toggle favorite
- `POST /sessions/<id>/tags` - Add tags
- `GET /sessions/<id>/export?format=<markdown|pdf|json>` - Export chat

### Settings
- `GET /settings` - Get user settings
- `POST /settings` - Update settings

## Customization

### Themes
- Retro (8-bit pixel style)
- Dark Mode
- Light Mode

### Settings Available
- Font size (12-18px)
- Difficulty level (Easy, Normal, Hard)
- Syntax highlighting toggle
- Auto-save toggle
- Ollama model selection

## Features in Detail

### Image Processing
- Upload images to include with messages
- Automatic OCR text extraction (if pytesseract available)
- Image metadata preservation

### Chat History
- Automatic session creation
- Messages grouped by session
- Auto-generated names from first message
- Search functionality
- Metadata: created date, updated date, message count

### Export Options
- **Markdown** - Formatted text with code blocks
- **PDF** - Professional formatted document
- **JSON** - Complete session data

##  Privacy

- All chat data stored locally in `chat_history.json`
- Settings stored in `user_settings.json`
- No external API calls beyond Ollama
- No data sent to external servers





---

**Made with ❤️ by RHENIU5**

Visit: https://github.com/RHENIU5/Retrobot
