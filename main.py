# First install all required packages - MINIMAL version
!pip install -U langchain langchain-community langchain-groq faiss-cpu pypdf gradio sentence-transformers


# Run the code
import os
import tempfile
from datetime import datetime
from google.colab import userdata
import gradio as gr

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Get API key
GROQ_API_KEY = userdata.get('GROQ_API_KEY')
if not GROQ_API_KEY:
    print("⚠️ Please set your GROQ_API_KEY in Google Colab secrets")
    print("Go to: Tools → Secrets → Add 'GROQ_API_KEY' with your key")
else:
    print("✅ API key loaded")

# Global variables
vector_store = None
user_name = ""
full_resume_text = ""

# ========== PROCESS RESUME ==========
def process_resume(file_obj):
    global vector_store, user_name, full_resume_text

    try:
        if file_obj is None:
            return "⚠️ Please select a PDF file!"

        # Save uploaded file
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name
        with open(file_obj, 'rb') as src, open(temp_path, 'wb') as dest:
            dest.write(src.read())

        # Load PDF
        print("📄 Loading resume...")
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Store full resume text
        full_resume_text = "\n\n".join([doc.page_content for doc in documents])

        # Extract name from resume (first line usually)
        lines = full_resume_text.split('\n')
        if lines:
            # Try to find name - usually first non-empty line
            for line in lines[:5]:  # Check first 5 lines
                if len(line.strip()) > 2 and not line.strip().isdigit():
                    user_name = line.strip()
                    break

        # Split and create embeddings
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        # Create vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)

        os.unlink(temp_path)

        preview = documents[0].page_content[:200] + "..."
        return f"✅ Resume loaded successfully!\n\nWelcome, {user_name if user_name else 'User'}! I'm ready to be your digital twin.\n\nPreview: {preview}"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ========== GET RESPONSE - DIGITAL TWIN STYLE ==========
def get_response(message):
    global vector_store, user_name, full_resume_text

    if vector_store is None:
        return "⚠️ Please upload your resume first in the 'Upload Resume' tab!"

    message_lower = message.lower()

    # ========== CALCULATOR ==========
    if "calculate" in message_lower or any(op in message for op in ["+", "-", "*", "/"]):
        try:
            import re
            # Find math expression
            expr_match = re.search(r'[\d\.\+\-\*\/\(\)]+', message)
            if expr_match:
                expr = expr_match.group()
                result = eval(expr)
                return f"🤖 Sure! Let me calculate that for you...\n\n📊 **Calculation**: {expr} = **{result}**"
        except:
            pass

    # ========== TIME ==========
    if "time" in message_lower or "clock" in message_lower or "current time" in message_lower:
        current_time = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        return f"🤖 Looking at my clock...\n\n⏰ **Current time**: {current_time}"

    # ========== GREETINGS ==========
    greetings = ["hi", "hello", "hey", "hola", "namaste", "good morning", "good afternoon", "good evening"]
    if any(greet in message_lower for greet in greetings):
        name_part = f", {user_name}" if user_name else ""
        return f"🤖 Hello there{name_part}! I'm your digital twin. 👋\n\nI can help you with:\n• Questions about your resume\n• Math calculations\n• Current time\n\nWhat would you like to know about yourself today?"

    # ========== CHECK IF QUESTION IS ABOUT RESUME ==========
    # List of allowed resume-related keywords
    resume_keywords = [
        "skill", "education", "project", "experience", "certification",
        "college", "university", "school", "degree", "course", "work",
        "background", "qualification", "achievement", "accomplishment",
        "what can", "what have", "tell me about my", "my", "i", "me"
    ]

    # Check if question is about the person
    is_about_me = any(keyword in message_lower for keyword in resume_keywords)

    # Also check if question starts with personal pronouns
    personal_starters = ["what are my", "tell me about my", "where did i", "when did i", "how did i", "do i have"]
    if not is_about_me:
        is_about_me = any(message_lower.startswith(starter) for starter in personal_starters)

    # If NOT about resume, give restricted response
    if not is_about_me:
        return "🤖 I'm your digital twin! I can only answer questions about YOU from your resume.\n\nTry asking:\n• What are my skills?\n• Tell me about my education\n• What projects have I worked on?\n• Calculate 25 * 4\n• What time is it?"

    # ========== SEARCH RESUME ==========
    try:
        # Search for more chunks to get complete information
        docs = vector_store.similarity_search(message, k=4)
        if docs:
            # Combine all relevant content
            resume_context = "\n\n".join([doc.page_content for doc in docs])
        else:
            # If no specific results, use general resume sections
            resume_context = full_resume_text[:800]
    except:
        resume_context = full_resume_text[:800]

    # Initialize LLM
    try:
        # Try different Groq models
        llm = ChatGroq(
            model="llama3-8b-8192",  # Use llama3 which should work
            temperature=0.3,
            api_key=GROQ_API_KEY,
            max_tokens=600
        )
    except Exception as e:
        return f"🤖 LLM Error: Could not initialize. Please check your API key.\nError: {str(e)}"

    # Create STRICT DIGITAL TWIN prompt
    prompt = f"""You are my digital twin - an AI version of me. You speak ONLY about me based on my resume.

STRICT RULES:
1. Speak in FIRST PERSON (use "I", "my", "me") as if you ARE me
2. Answer ONLY based on the resume information below
3. If something is not in the resume, say: "That's not in my resume. Please ask about my skills, education, or projects."
4. Give COMPLETE answers - don't cut off mid-sentence
5. Use bullet points or lists for clarity
6. Add relevant emojis
7. NEVER give general knowledge or information not in resume

MY RESUME INFORMATION:
{resume_context}

USER QUESTION: {message}

IMPORTANT: If the question asks for a list (like skills or projects), list ALL of them completely.

NOW RESPOND AS MY DIGITAL TWIN:"""

    try:
        response = llm.invoke(prompt)
        response_text = response.content

        # Ensure response is complete
        if not response_text.endswith(('.', '!', '?', ':')) and len(response_text) > 50:
            response_text += "."

        return response_text

    except Exception as e:
        print(f"LLM Error: {e}")
        # Direct fallback: extract relevant lines from resume
        lines = full_resume_text.split('\n')
        relevant_lines = []

        # Find lines related to the question
        query_words = message_lower.split()
        for line in lines:
            line_lower = line.lower()
            # Check if any query word is in line (excluding common words)
            common_words = ['what', 'are', 'my', 'the', 'and', 'for', 'with', 'about']
            query_words_filtered = [w for w in query_words if w not in common_words]

            if any(word in line_lower for word in query_words_filtered) and len(line.strip()) > 10:
                relevant_lines.append(line.strip())

        if relevant_lines:
            return f"🤖 Based on my resume:\n\n" + "\n".join(relevant_lines[:8])
        else:
            return "🤖 That information is not in my resume. Please ask about my skills, education, or projects."

# ========== GRADIO INTERFACE ==========
with gr.Blocks(title="🤖 Digital Twin") as demo:
    gr.Markdown("# 🤖 My Digital Twin")
    gr.Markdown("### Chat with an AI version of yourself - ONLY from your resume")

    with gr.Tab("📄 Step 1: Upload Resume"):
        gr.Markdown("Upload your resume so I can become your digital twin!")
        file_input = gr.File(
            label="Upload your PDF resume",
            file_types=[".pdf"],
            type="filepath"
        )
        upload_btn = gr.Button("🚀 Create My Digital Twin", variant="primary")
        status_output = gr.Textbox(label="Status", lines=5)

        upload_btn.click(
            fn=process_resume,
            inputs=[file_input],
            outputs=[status_output]
        )

    with gr.Tab("💬 Step 2: Chat with Me"):
        chatbot = gr.Chatbot(
            label="Chat with your digital twin",
            height=400,
        )
        msg = gr.Textbox(
            label="Ask about YOURSELF from your resume",
            placeholder="Examples: What are my skills? Tell me about my projects...",
            scale=4
        )
        clear_btn = gr.Button("🗑️ Clear Chat")

        def user(message, history):
            """When user sends message"""
            if vector_store is None:
                return history + [[message, "🤖 Hi there! 👋 Please upload your resume in Step 1 first."]]
            return history + [[message, None]]

        def bot(history):
            """Get bot response"""
            if not history or history[-1][1] is not None:
                return history

            user_message = history[-1][0]
            bot_response = get_response(user_message)
            history[-1][1] = bot_response
            return history

        # When user presses Enter
        msg.submit(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[chatbot],
            queue=False
        ).then(
            fn=bot,
            inputs=[chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "",  # Clear input box
            None,
            [msg]
        )

        # Clear chat
        clear_btn.click(
            lambda: None,
            None,
            [chatbot],
            queue=False
        )

        # Examples - ONLY resume-related
        gr.Examples(
            examples=[
                "What are ALL my technical skills?",
                "Tell me COMPLETELY about my education",
                "What projects have I worked on? List all",
                "What certifications do I have?",
                "Calculate 25 * 4",
                "What time is it?"
            ],
            inputs=[msg],
            label="Click to ask about yourself:"
        )

    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## Your Digital Twin - Resume Only

        **I ONLY answer from your resume!**

        ✅ **I can answer:**
        - Questions about YOUR skills
        - YOUR education background
        - YOUR projects and experience
        - Simple calculations
        - Current time

        ❌ **I WON'T answer:**
        - General knowledge questions
        - Questions about other people
        - Information not in your resume

        **Perfect for:** Interview practice, self-reflection, resume review
        """)

print("\n" + "="*60)
print("🚀 STRICT DIGITAL TWIN - RESUME ONLY")
print("="*60)
print("1. Upload your resume")
print("2. Ask ONLY about yourself")
print("3. Get COMPLETE responses from resume")
print("="*60)

# Launch the app
try:
    demo.launch(share=True, debug=False, theme=gr.themes.Soft())
except Exception as e:
    print(f"Launch error: {e}")
    demo.launch(debug=False)
