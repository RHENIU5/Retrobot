"""
AI Chatbot Web Application with Enhanced Features
Backend: Flask
AI: Ollama (Local LLM)
Features: Chat modes, image processing, user settings, code formatting, chat exports
"""

from flask import Flask, render_template, request, jsonify, send_file
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import json
from datetime import datetime
import uuid
import re
import base64
from io import BytesIO
import markdown
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

# Optional: Image processing
try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

# Optional: OCR
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI client with Ollama API
# Ollama provides an OpenAI-compatible API endpoint
endpoint = os.getenv("ollama_endpoint", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

print(f"[Ollama Config]")
print(f"  Endpoint: {endpoint}")
print(f"  Model: {ollama_model}")

try:
    client = OpenAI(
        base_url=endpoint + "/v1",
        api_key="ollama"
    )
except Exception as e:
    print(f"[Warning] Failed to initialize Ollama client: {e}")
    client = None

# Chat history persistence with JSON
HISTORY_FILE = Path(__file__).parent / "chat_history.json"
USER_SETTINGS_FILE = Path(__file__).parent / "user_settings.json"
conversation_history = []
current_session_id = str(uuid.uuid4())
current_user_settings = {}

def load_user_settings():
    """Load user preferences and customization settings"""
    global current_user_settings
    if USER_SETTINGS_FILE.exists():
        try:
            with open(USER_SETTINGS_FILE, 'r') as f:
                current_user_settings = json.load(f)
        except (json.JSONDecodeError, IOError):
            current_user_settings = get_default_settings()
    else:
        current_user_settings = get_default_settings()
    return current_user_settings

def get_default_settings():
    """Get default user settings"""
    return {
        "theme": "retro",  # retro, dark, light
        "font_size": 14,   # pixels
        "difficulty": "normal",  # easy, normal, hard
        "syntax_highlighting": True,
        "code_execution": False,
        "auto_save": True,
        "language": "en"
    }

def save_user_settings():
    """Save user settings to JSON file"""
    try:
        with open(USER_SETTINGS_FILE, 'w') as f:
            json.dump(current_user_settings, f, indent=2)
    except IOError as e:
        print(f"Error saving user settings: {e}")

def load_history(session_id=None):
    """Load chat history from JSON file for a specific session or create new"""
    global conversation_history, current_session_id
    
    if session_id:
        current_session_id = session_id
    
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
                if current_session_id in data:
                    conversation_history = data[current_session_id].get("messages", [])
                else:
                    conversation_history = []
        except (json.JSONDecodeError, IOError):
            conversation_history = []
    else:
        conversation_history = []

def save_history():
    """Save current conversation to JSON file"""
    global conversation_history, current_session_id
    
    # Load existing history
    data = {}
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {}
    
    # Update current session
    if current_session_id not in data:
        data[current_session_id] = {
            "created": datetime.now().isoformat(),
            "messages": [],
            "name": None  # Auto-generated name will be created from first message
        }
    
    data[current_session_id]["messages"] = conversation_history
    data[current_session_id]["updated"] = datetime.now().isoformat()
    
    # Auto-generate name from first user message if not set
    if not data[current_session_id].get("name"):
        for msg in conversation_history:
            if msg.get("role") == "user":
                first_message = msg.get("content", "").strip()[:50]
                if first_message:
                    data[current_session_id]["name"] = first_message
                break
    
    # Save to file
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        print(f"Error saving history: {e}")

def get_all_sessions():
    """Get all conversation sessions"""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
                return {
                    session_id: {
                        "name": sess.get("name") or "Untitled Chat",
                        "created": sess.get("created"),
                        "updated": sess.get("updated"),
                        "message_count": len(sess.get("messages", []))
                    }
                    for session_id, sess in data.items()
                }
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

# Load initial history
load_history()

# Load user settings
load_user_settings()

# System prompt for the AI assistant
CODE_SYSTEM_PROMPT = """You are an expert code generation AI. ONLY output code in markdown blocks.

CRITICAL RULES:
1. Output code IMMEDIATELY in ```language code blocks
2. NEVER write "CODEBLOCK0", "BLOCK0", or any placeholder text
3. NEVER explain before code - output code FIRST
4. Output COMPLETE, WORKING, EXECUTABLE code
5. Include ALL required imports, class definitions, and functions
6. Add proper indentation and formatting
7. After code block (```), add ONE brief explanation line (1-2 sentences)
8. FOCUS ONLY on what the user asked for - ignore unrelated topics
9. NO extra information or off-topic content

STOP IMMEDIATELY after providing the code and explanation."""

RESEARCH_SYSTEM_PROMPT = """You are an expert research assistant. Provide focused, accurate information.

CRITICAL RULES:
1. FOCUS ONLY on what the user asked for
2. Answer the specific question directly and completely
3. Use clear headings for different sections
4. Provide authoritative, accurate information
5. Use bullet points for lists
6. NO off-topic content or unrelated information
7. Keep response organized and easy to read
8. Be concise but comprehensive
9. If question is unclear, ask for clarification instead of guessing

ANSWER ONLY what was asked. NOTHING ELSE."""

CHAT_SYSTEM_PROMPT = """You are Retrobot, a helpful, friendly AI assistant.

CRITICAL RULES:
1. ANSWER ONLY what the user asked - no extra content
2. Be conversational and warm
3. Answer questions clearly and concisely
4. Ask clarifying questions if needed
5. Provide helpful follow-up suggestions only when relevant
6. Use proper grammar and punctuation
7. Match the user's tone appropriately
8. NO information mixing or off-topic responses
9. Format response clearly with proper line breaks

FOCUS: Answer the specific question asked. Nothing more, nothing less."""

# NEW SYSTEM PROMPTS FOR ENHANCED MODES

TUTORIAL_SYSTEM_PROMPT = """You are an expert educator and tutorial writer.

CRITICAL RULES:
1. Provide step-by-step instructions
2. Use numbered lists for sequential steps
3. Include explanations for each step
4. Provide code examples where applicable
5. Add tips and common mistakes to avoid
6. Use clear headings and subheadings
7. Focus on learning and understanding
8. Adjust complexity level based on user request
9. Provide practice exercises when relevant

FORMAT:
1. Start with a brief overview
2. Break down into logical steps
3. Add examples for each step
4. End with practice or review questions"""

WRITING_SYSTEM_PROMPT = """You are an expert writing assistant and editor.

CRITICAL RULES:
1. Provide writing samples or templates
2. Match the requested tone (formal, casual, creative)
3. Maintain consistency and clarity
4. Use proper grammar and punctuation
5. Provide alternative phrasings
6. Include writing tips and best practices
7. Consider the target audience
8. Format with proper paragraph breaks
9. Offer revision suggestions

SUPPORT: emails, essays, articles, stories, proposals"""

CREATIVE_SYSTEM_PROMPT = """You are an imaginative creative assistant.

CRITICAL RULES:
1. Generate creative ideas and suggestions
2. Think outside the box
3. Provide multiple options and variations
4. Include brainstorming lists
5. Add unexpected combinations and ideas
6. Use vivid, descriptive language
7. Build on user ideas with new perspectives
8. Keep energy and enthusiasm high
9. Format ideas in organized lists

FOCUS: Creativity, innovation, brainstorming, ideation"""

DEBUGGING_SYSTEM_PROMPT = """You are an expert debugging assistant.

CRITICAL RULES:
1. Identify and explain the error
2. Provide root cause analysis
3. Suggest multiple solutions
4. Include fixed code examples
5. Explain what changed and why
6. Add prevention tips
7. Test cases for verification
8. Performance considerations
9. Format code in proper blocks

APPROACH: Problem → Analysis → Solutions → Prevention"""

def detect_query_type(text):
    """Detect the type of query to use appropriate system prompt and formatting"""
    text_lower = text.lower()
    
    # Coding keywords
    coding_keywords = [
        'code', 'write', 'function', 'class', 'method', 'script', 'program',
        'algorithm', 'debug', 'error', 'bug', 'fix', 'implement', 'create',
        'javascript', 'python', 'java', 'c++', 'cpp', 'go', 'rust', 'typescript',
        'react', 'vue', 'angular', 'django', 'flask', 'express', 'node',
        'database', 'sql', 'api', 'rest', 'json', 'xml', 'html', 'css',
        'regex', 'loop', 'array', 'string', 'variable', 'conditional',
        'syntax', 'library', 'framework', 'module', 'package', 'import',
        'def ', 'function ', 'class ', '{', '}', '(', ')', '=>'
    ]
    
    # Research keywords
    research_keywords = [
        'research', 'study', 'analyze', 'analyze', 'explain', 'how', 'what is', 'why',
        'history', 'background', 'overview', 'summary', 'report', 'information',
        'tell me about', 'describe', 'definition', 'concept', 'theory',
        'statistics', 'data', 'trend', 'comparison', 'difference between'
    ]
    
    # Tutorial/Learning keywords
    tutorial_keywords = [
        'teach', 'learn', 'tutorial', 'step by step', 'how to', 'guide', 'beginner',
        'help me understand', 'explain step by step', 'show me', 'lesson', 'course',
        'introduction to', 'basics of', 'get started', 'hands on', 'practice',
        'exercise', '101', 'for dummies'
    ]
    
    # Writing keywords
    writing_keywords = [
        'write', 'essay', 'email', 'letter', 'article', 'blog', 'story', 'poem',
        'write me', 'compose', 'draft', 'rewrite', 'edit', 'proofread', 'grammar',
        'tone', 'formal', 'casual', 'persuasive', 'narrative', 'description',
        'proposal', 'pitch', 'cover letter', 'resume'
    ]
    
    # Creative keywords
    creative_keywords = [
        'brainstorm', 'ideas', 'creative', 'imagine', 'design', 'invent', 'think of',
        'suggest', 'recommendations', 'possibilities', 'alternatives', 'variations',
        'concept', 'inspire', 'innovation', 'unique', 'fresh', 'what if', 'generate ideas'
    ]
    
    # Debugging keywords
    debugging_keywords = [
        'debug', 'fix', 'error', 'bug', 'problem', 'issue', 'broken', 'crash',
        'not working', 'exception', 'trace', 'stack trace', 'why is', 'why does',
        'help debug', 'troubleshoot', 'fix this', 'failing', 'doesn\'t work'
    ]
    
    # Check each type (order matters for specificity)
    if any(keyword in text_lower for keyword in debugging_keywords):
        return 'debugging'
    
    if any(keyword in text_lower for keyword in creative_keywords):
        return 'creative'
    
    if any(keyword in text_lower for keyword in writing_keywords):
        return 'writing'
    
    if any(keyword in text_lower for keyword in tutorial_keywords):
        return 'tutorial'
    
    if any(keyword in text_lower for keyword in coding_keywords):
        return 'coding'
    
    if any(keyword in text_lower for keyword in research_keywords):
        return 'research'
    
    return 'chat'

def extract_language(text):
    """Extract programming language from user request"""
    languages = {
        'python': ['python', 'py'],
        'java': ['java'],
        'javascript': ['javascript', 'js', 'node'],
        'cpp': ['c++', 'cpp'],
        'go': ['go', 'golang'],
        'rust': ['rust'],
        'php': ['php'],
        'ruby': ['ruby'],
        'typescript': ['typescript', 'ts'],
        'csharp': ['c#', 'csharp']
    }
    
    text_lower = text.lower()
    for lang, keywords in languages.items():
        for keyword in keywords:
            if keyword in text_lower:
                return lang
    return 'java'  # default

# ============================================================
# IMAGE PROCESSING FUNCTIONS (Feature 2)
# ============================================================

def process_image(base64_data, image_name):
    """Process image and extract metadata/text"""
    try:
        # Decode base64
        image_data = base64.b64decode(base64_data)
        
        # Load image
        if PILLOW_AVAILABLE:
            image = Image.open(BytesIO(image_data))
            image_info = {
                "name": image_name,
                "size": f"{image.width}x{image.height}",
                "format": image.format,
                "mode": image.mode,
            }
            
            # Try OCR if available
            if PYTESSERACT_AVAILABLE:
                try:
                    extracted_text = pytesseract.image_to_string(image)
                    if extracted_text.strip():
                        image_info["text"] = extracted_text
                except Exception as e:
                    print(f"[OCR Warning] Could not extract text from image: {e}")
            
            return image_info
        else:
            # No PIL, return basic info
            return {
                "name": image_name,
                "size": len(image_data),
                "status": "Image received (PIL not available for full processing)"
            }
    
    except Exception as e:
        print(f"[Image Processing Error] {e}")
        return {"name": image_name, "error": str(e)}

def generate_image_description(image_info):
    """Generate description for AI to understand the image"""
    description = f"Image '{image_info.get('name')}': "
    
    if "error" in image_info:
        description += f"Could not process image. Error: {image_info['error']}"
    else:
        description += f"Size: {image_info.get('size', 'unknown')}, Format: {image_info.get('format', 'unknown')}"
        
        if "text" in image_info:
            description += f"\nText in image:\n{image_info['text'][:200]}..."
    
    return description

# ============================================================
# EXPORT AND FORMATTING FUNCTIONS (Features 3, 4, 5)
# ============================================================

def format_code_block(content, language=""):
    """Format code with syntax highlighting metadata"""
    # Detect language if not provided
    if not language and any(lang in content.lower() for lang in ['python', 'import', 'def ', 'class ']):
        language = 'python'
    
    return f"""```{language}
{content}
```"""

def export_session_to_markdown(session_id, session_data):
    """Export session as Markdown file"""
    md_content = f"# Chat Session Export\n\n"
    md_content += f"**Date Created:** {session_data.get('created', 'Unknown')}\n\n"
    md_content += f"**Tags:** {', '.join(session_data.get('tags', []))}\n\n"
    md_content += f"**Favorite:** {'Yes' if session_data.get('is_favorite', False) else 'No'}\n\n"
    md_content += f"---\n\n"
    
    for message in session_data.get('messages', []):
        role = message.get('role', 'unknown').upper()
        content = message.get('content', '')
        md_content += f"**{role}:**\n\n{content}\n\n---\n\n"
    
    return md_content

def export_session_to_pdf(session_id, session_data):
    """Export session as PDF file"""
    try:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Add header
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='#4a90e2',
            spaceAfter=30
        )
        story.append(Paragraph("Chat Session Export", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Add metadata
        meta_style = styles['Normal']
        story.append(Paragraph(f"<b>Date:</b> {session_data.get('created', 'Unknown')}", meta_style))
        story.append(Paragraph(f"<b>Tags:</b> {', '.join(session_data.get('tags', []))}", meta_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Add messages
        for message in session_data.get('messages', []):
            role = message.get('role', 'unknown').upper()
            content = message.get('content', '')
            
            # Escape HTML special characters
            content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            msg_style = ParagraphStyle(
                'Message',
                parent=styles['Normal'],
                textColor='#333333' if role == 'USER' else '#666666',
                spaceAfter=12
            )
            
            story.append(Paragraph(f"<b>{role}:</b> {content[:200]}...", msg_style))
            story.append(Spacer(1, 0.1*inch))
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer
    
    except Exception as e:
        print(f"[PDF Export Error] {e}")
        return None

def add_session_metadata(session_data):
    """Add metadata fields to session for enhanced history tracking"""
    if 'tags' not in session_data:
        session_data['tags'] = []
    if 'is_favorite' not in session_data:
        session_data['is_favorite'] = False
    if 'query_types' not in session_data:
        session_data['query_types'] = []
    if 'user_settings' not in session_data:
        session_data['user_settings'] = {}
    
    return session_data

def clean_response(response_text, query_type):
    """Clean and format response to ensure focus and professionalism"""
    
    # Remove common unrelated patterns
    patterns_to_remove = [
        r'Answer:.*?\n',  # Remove "Answer:" prefix if repeating the question
        r'Follow-up:.*',  # Remove follow-up suggestions unless needed
        r'Note:.*',  # Remove random notes
        r'\[.*?\]',  # Remove brackets with extra info
    ]
    
    cleaned = response_text
    
    # For code responses, keep only code blocks and brief explanation
    if query_type == 'coding':
        # Extract code blocks and explanation
        code_match = re.search(r'```[\s\S]*?```', cleaned)
        if code_match:
            code_block = code_match.group(0)
            # Get explanation after code (single sentence)
            explanation_match = re.search(r'```\s*\n\n?(.*?)(?:\n|$)', cleaned)
            explanation = explanation_match.group(1).strip() if explanation_match else ''
            
            # Keep only first sentence of explanation
            first_sentence = explanation.split('\n')[0][:150]
            
            cleaned = code_block + '\n' + first_sentence
    
    # For research, remove duplicate information
    elif query_type == 'research':
        # Split by double newlines and keep unique paragraphs
        paragraphs = cleaned.split('\n\n')
        seen = set()
        unique_paragraphs = []
        for para in paragraphs:
            # Use first 50 chars as identifier to avoid exact duplicates
            para_id = para[:50].lower()
            if para_id not in seen and len(para.strip()) > 10:
                seen.add(para_id)
                unique_paragraphs.append(para)
        
        cleaned = '\n\n'.join(unique_paragraphs)
    
    # Remove trailing whitespace and excessive newlines
    cleaned = cleaned.strip()
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned

def generate_fallback_code(user_message):
    """Generate working code for common requests if model fails"""
    text_lower = user_message.lower()
    lang = extract_language(user_message)
    
    # Detect "add two numbers" request
    if any(word in text_lower for word in ['add', 'sum', '+']) and any(word in text_lower for word in ['two', '2', 'number', 'int']):
        if lang == 'python':
            return '''```python
def add_numbers(a, b):
    """Add two numbers and return the sum"""
    return a + b

# Example usage
num1 = 5
num2 = 3
result = add_numbers(num1, num2)
print(f"The sum of {num1} and {num2} is {result}")
```
This function adds two numbers together using basic arithmetic.'''
        elif lang == 'javascript':
            return '''```javascript
function addNumbers(a, b) {
    // Add two numbers and return the sum
    return a + b;
}

// Example usage
let num1 = 5;
let num2 = 3;
let result = addNumbers(num1, num2);
console.log(`The sum of ${num1} and ${num2} is ${result}`);
```
This function adds two numbers together using basic arithmetic.'''
        elif lang == 'cpp':
            return '''```cpp
#include <iostream>
using namespace std;

int add(int a, int b) {
    return a + b;
}

int main() {
    int num1 = 5;
    int num2 = 3;
    cout << "Sum: " << add(num1, num2) << endl;
    return 0;
}
```
This program adds two numbers and displays the result.'''
        else:  # java (default)
            return '''```java
public class AddNumbers {
    public static int add(int a, int b) {
        return a + b;
    }

    public static void main(String[] args) {
        int num1 = 5;
        int num2 = 3;
        int sum = add(num1, num2);
        System.out.println("The sum of " + num1 + " and " + num2 + " is " + sum);
    }
}
```
This Java program adds two numbers together and prints the result.'''
    
    return None

def fix_code_response(response_text, user_message):
    """Check if response has placeholder and generate actual code"""
    # Check for placeholders
    placeholders = ['CODEBLOCK0', 'BLOCK0', 'BLOCK1', 'CODE HERE', 'CODE BLOCK', '[code]']
    has_placeholder = any(placeholder in response_text for placeholder in placeholders)
    
    if has_placeholder:
        print(f"[Warning] Detected placeholder in response, generating actual code...")
        fallback = generate_fallback_code(user_message)
        if fallback:
            return fallback
    
    return response_text

@app.route("/")
def index():
    """
    Home route - renders the main chatbot interface
    """
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat endpoint - receives user message and returns AI response
    Enhanced with: chat modes, image processing, user settings
    Expects: JSON {message: "user message", images: [...]}
    Returns: JSON {reply: "ai response", query_type: "detected_type"}
    """
    try:
        # Check if client is initialized
        if client is None:
            return jsonify({
                "error": "Ollama service not available. Please ensure Ollama is running on " + endpoint
            }), 503
        
        # Get the JSON data from request
        data = request.get_json()
        
        # Validate that message or images exist
        if not data or (not data.get("message") and not data.get("images")):
            return jsonify({"error": "Message or images required"}), 400
        
        user_message = data.get("message", "").strip()
        images = data.get("images", [])
        
        # FEATURE 2: Image Processing
        image_descriptions = []
        if images:
            for img in images:
                if isinstance(img, dict) and 'data' in img:
                    try:
                        base64_data = img['data'].split(',')[1] if ',' in img['data'] else img['data']
                        image_info = process_image(base64_data, img.get('name', 'image'))
                        image_desc = generate_image_description(image_info)
                        image_descriptions.append(image_desc)
                    except Exception as e:
                        image_descriptions.append(f"Image '{img.get('name', 'unknown')}': Error processing - {e}")
        
        # Build user content with image descriptions
        user_content = user_message
        if image_descriptions:
            image_text = "\n".join(image_descriptions)
            user_content = f"{image_text}\n\n{user_message}" if user_message else image_text
        
        if not user_content.strip():
            return jsonify({"error": "Message or images cannot be empty"}), 400
        
        # Add user message to conversation history
        conversation_history.append({
            "role": "user",
            "content": user_content
        })
        
        # FEATURE 1: Enhanced Chat Mode Detection
        query_type = detect_query_type(user_message)
        
        # Select system prompt and parameters based on query type
        system_prompts = {
            'coding': (CODE_SYSTEM_PROMPT, 0.1, 2000),
            'research': (RESEARCH_SYSTEM_PROMPT, 0.5, 1500),
            'tutorial': (TUTORIAL_SYSTEM_PROMPT, 0.5, 1500),
            'writing': (WRITING_SYSTEM_PROMPT, 0.6, 1500),
            'creative': (CREATIVE_SYSTEM_PROMPT, 0.8, 1500),
            'debugging': (DEBUGGING_SYSTEM_PROMPT, 0.2, 2000),
            'chat': (CHAT_SYSTEM_PROMPT, 0.7, 800),
        }
        
        system_prompt, temperature, max_tokens = system_prompts.get(query_type, system_prompts['chat'])
        
        # Build messages list for API call
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        
        # Call Ollama API
        response = client.chat.completions.create(
            model=ollama_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract assistant response
        assistant_message = response.choices[0].message.content
        
        # FEATURE 4: Code-specific formatting
        if query_type == 'coding':
            assistant_message = fix_code_response(assistant_message, user_message)
        
        # Clean and format response
        assistant_message = clean_response(assistant_message, query_type)
        
        # Add assistant response to conversation history
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message,
            "query_type": query_type  # Store the detected query type
        })
        
        # Save history to JSON file
        save_history()
        
        # Return response with query type information
        return jsonify({
            "reply": assistant_message,
            "query_type": query_type,
            "image_count": len(images),
            "settings": current_user_settings
        })
    
    except Exception as e:
        # Proper error handling with detailed messages
        error_str = str(e)
        print(f"[Error] Chat endpoint error: {error_str}")
        
        # Check for connection errors
        if "Connection refused" in error_str or "Failed to resolve" in error_str or "HTTPConnectionPool" in error_str:
            error_message = f"Cannot connect to Ollama at {endpoint}. Please ensure Ollama is running with 'ollama serve'"
        # Check for model not found errors
        elif "model" in error_str.lower() and "not found" in error_str.lower():
            error_message = f"Model '{ollama_model}' not found. Please install it with: ollama pull {ollama_model}"
        # Check for API errors
        elif "401" in error_str or "Unauthorized" in error_str:
            error_message = "Authentication Error: Invalid API key."
        elif "429" in error_str:
            error_message = "Rate Limit: Too many requests. Please wait a moment."
        elif "500" in error_str or "502" in error_str or "503" in error_str:
            error_message = "Ollama server error. Please check the Ollama service."
        else:
            error_message = f"Error: {error_str[:200]}"
        
        return jsonify({"error": error_message}), 500

@app.route("/clear", methods=["POST"])
def clear_chat():
    """
    Clear chat history endpoint - starts a new session
    """
    global conversation_history, current_session_id
    conversation_history = []
    current_session_id = str(uuid.uuid4())
    save_history()
    return jsonify({"status": "Chat cleared", "session_id": current_session_id})

@app.route("/sessions", methods=["GET"])
def get_sessions():
    """
    Get all conversation sessions with metadata
    """
    sessions = get_all_sessions()
    return jsonify({"sessions": sessions})

@app.route("/sessions/<session_id>", methods=["GET"])
def load_session(session_id):
    """
    Load a specific conversation session
    """
    global conversation_history, current_session_id
    load_history(session_id)
    return jsonify({
        "session_id": session_id,
        "messages": conversation_history,
        "message_count": len(conversation_history)
    })

@app.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """
    Delete a specific conversation session
    """
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
            
            if session_id in data:
                del data[session_id]
                with open(HISTORY_FILE, 'w') as f:
                    json.dump(data, f, indent=2)
                return jsonify({"status": f"Session {session_id} deleted"})
            else:
                return jsonify({"error": "Session not found"}), 404
        except (json.JSONDecodeError, IOError) as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "No history file found"}), 404

@app.route("/sessions/<session_id>", methods=["PUT"])
def rename_session(session_id):
    """
    Rename a conversation session
    Expects: JSON {name: "new session name"}
    """
    try:
        data_req = request.get_json()
        
        if not data_req or "name" not in data_req:
            return jsonify({"error": "Name field required"}), 400
        
        new_name = data_req["name"].strip()
        
        if not new_name:
            return jsonify({"error": "Name cannot be empty"}), 400
        
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
            
            if session_id in data:
                data[session_id]["name"] = new_name
                with open(HISTORY_FILE, 'w') as f:
                    json.dump(data, f, indent=2)
                return jsonify({
                    "status": "Session renamed",
                    "session_id": session_id,
                    "name": new_name
                })
            else:
                return jsonify({"error": "Session not found"}), 404
        else:
            return jsonify({"error": "No history file found"}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# NEW ENDPOINTS FOR ENHANCED FEATURES
# ============================================================

# FEATURE 5: User Settings Management
@app.route("/settings", methods=["GET"])
def get_settings():
    """Get current user settings"""
    return jsonify(current_user_settings)

@app.route("/settings", methods=["POST"])
def update_settings():
    """Update user settings"""
    global current_user_settings
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No settings provided"}), 400
        
        # Update only provided settings
        for key, value in data.items():
            if key in current_user_settings:
                current_user_settings[key] = value
        
        save_user_settings()
        return jsonify({
            "status": "Settings updated",
            "settings": current_user_settings
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# FEATURE 3: Session Tagging and Favorites
@app.route("/sessions/<session_id>/tags", methods=["POST"])
def add_tags(session_id):
    """Add tags to a session"""
    try:
        data = request.get_json()
        tags = data.get("tags", []) if data else []
        
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                history_data = json.load(f)
            
            if session_id in history_data:
                if 'tags' not in history_data[session_id]:
                    history_data[session_id]['tags'] = []
                
                # Add new tags (avoid duplicates)
                for tag in tags:
                    if tag not in history_data[session_id]['tags']:
                        history_data[session_id]['tags'].append(tag)
                
                with open(HISTORY_FILE, 'w') as f:
                    json.dump(history_data, f, indent=2)
                
                return jsonify({
                    "status": "Tags added",
                    "session_id": session_id,
                    "tags": history_data[session_id]['tags']
                })
            else:
                return jsonify({"error": "Session not found"}), 404
        return jsonify({"error": "No history found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/sessions/<session_id>/favorite", methods=["POST"])
def toggle_favorite(session_id):
    """Mark session as favorite or remove favorite status"""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                history_data = json.load(f)
            
            if session_id in history_data:
                is_favorite = history_data[session_id].get('is_favorite', False)
                history_data[session_id]['is_favorite'] = not is_favorite
                
                with open(HISTORY_FILE, 'w') as f:
                    json.dump(history_data, f, indent=2)
                
                return jsonify({
                    "status": "Favorite toggled",
                    "session_id": session_id,
                    "is_favorite": history_data[session_id]['is_favorite']
                })
            else:
                return jsonify({"error": "Session not found"}), 404
        return jsonify({"error": "No history found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# FEATURE 3: Session Export
@app.route("/sessions/<session_id>/export", methods=["GET"])
def export_session(session_id):
    """Export session in specified format (markdown or pdf)"""
    try:
        export_format = request.args.get('format', 'markdown').lower()
        
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                history_data = json.load(f)
            
            if session_id not in history_data:
                return jsonify({"error": "Session not found"}), 404
            
            session_data = history_data[session_id]
            
            if export_format == 'pdf':
                pdf_buffer = export_session_to_pdf(session_id, session_data)
                if pdf_buffer:
                    return send_file(
                        pdf_buffer,
                        mimetype='application/pdf',
                        as_attachment=True,
                        download_name=f"chat_export_{session_id[:8]}.pdf"
                    )
                else:
                    return jsonify({"error": "Failed to generate PDF"}), 500
            
            else:  # markdown (default)
                md_content = export_session_to_markdown(session_id, session_data)
                return send_file(
                    BytesIO(md_content.encode('utf-8')),
                    mimetype='text/markdown',
                    as_attachment=True,
                    download_name=f"chat_export_{session_id[:8]}.md"
                )
        
        return jsonify({"error": "No history found"}), 404
    
    except Exception as e:
        print(f"[Export Error] {e}")
        return jsonify({"error": str(e)}), 500

# FEATURE 3: Search and Filter Sessions
@app.route("/sessions/search", methods=["GET"])
def search_sessions():
    """Search sessions by name, tags, or date range"""
    try:
        query = request.args.get('q', '').lower()
        tag_filter = request.args.get('tag', '')
        favorites_only = request.args.get('favorites', 'false').lower() == 'true'
        
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                history_data = json.load(f)
            
            results = {}
            for session_id, session_data in history_data.items():
                # Apply filters
                if favorites_only and not session_data.get('is_favorite', False):
                    continue
                
                if tag_filter and tag_filter not in session_data.get('tags', []):
                    continue
                
                if query:
                    name = session_data.get('name', '').lower()
                    messages_text = ' '.join([m.get('content', '').lower() for m in session_data.get('messages', [])])
                    
                    if query not in name and query not in messages_text:
                        continue
                
                results[session_id] = {
                    "name": session_data.get("name", "Untitled"),
                    "created": session_data.get("created"),
                    "tags": session_data.get("tags", []),
                    "is_favorite": session_data.get("is_favorite", False),
                    "message_count": len(session_data.get("messages", []))
                }
            
            return jsonify({
                "results": results,
                "count": len(results),
                "query": query,
                "tag_filter": tag_filter,
                "favorites_only": favorites_only
            })
        
        return jsonify({"error": "No history found"}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask development server on port 8000
    print("\n" + "="*50)
    print("🤖 Retrobot AI Assistant")
    print("="*50)
    print(f"Starting server: http://127.0.0.1:8000")
    print(f"Ollama Endpoint: {endpoint}")
    print(f"Model: {ollama_model}")
    print("="*50 + "\n")
    
    app.run(debug=True, host="127.0.0.1", port=8000)
