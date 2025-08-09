# Real Estate AI Assistant

An intelligent agent that answers complex business questions by searching a private knowledge base of internal documents, emails, and chats.

## Key Features

*   **Advanced RAG**: Uses a fine-tuned model and re-ranking for highly accurate search.
*   **Intelligent Agent**: Leverages LangChain to reason and use specialized tools for qualitative and quantitative analysis.
*   **Data Ingestion**: Processes `.docx`, `.txt`, and `.jsonl` files into a FAISS vector store.
*   **Interactive UI**: A user-friendly chat interface built with Streamlit.

## How to Use

### 1. Setup

First, clone the repository and set up your environment.

# Clone the repository
git clone https://github.com/venkatsr2/RE_assistant.git
cd RE_assistant

# Install dependencies
pip install -r requirements.txt

# Create and configure your environment file
cp .env.example .env

### Next, add your OPENAI_API_KEY to the newly created .env file.
### 2. Process Your Data
### Place all your source documents into the /data directory. Then, run the processing script to create the knowledge base. This only needs to be done once or when your documents change.

python process_data.py

### 3. Run the Application
### You can interact with the agent via the command line or the web interface.

### Option A: Web App (Recommended)

streamlit run streamlit_app.py

### Navigate to http://localhost:8501 in your browser.

### Option B: Command-Line (For Debugging)

python agent_pro.py

# Technology Stack
- **AI & Orchestration**: LangChain, OpenAI
- **NLP & Embeddings**: SentenceTransformers, Hugging Face
- **Vector Database**: FAISS
- **Data Handling**: Pandas
- **Web Interface**: Streamlit