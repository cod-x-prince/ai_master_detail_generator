# app.py (Final, Truly Free, and Stable Version with Local Embeddings)

# --- FIX for Asynchronous Conflict in Streamlit ---
import nest_asyncio
nest_asyncio.apply()
# ----------------------------------------------------

import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

# Import all necessary model providers and components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings # ✅ The stable, local embedding model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.memory import ConversationBufferMemory

# Document Processing Libraries
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
st.set_page_config(layout="wide")

# --- UI ENHANCEMENT: Custom CSS for Smooth Animations ---
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stChatMessage { animation: fadeIn 0.5s ease-out; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Trio AI Research Assistant")
st.caption("Now with Document Analysis (PDF/Word) and Full Conversation Memory.")

# --- 2. MODEL INITIALIZATION ---
try:
    # ✅ STRATEGIC CHANGE: Switched to a local Hugging Face model for embeddings.
    # This is 100% free, runs on your machine, and avoids all API rate limits.
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Researcher Models
    model_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
    # ✅ MAINTENANCE UPDATE: Upgraded to the latest Llama 3.3 model on Groq
    model_groq_researcher = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    # Using the stable ChatOpenAI client to connect to OpenRouter's compatible API
    model_openrouter = ChatOpenAI(
        model="mistralai/mistral-7b-instruct:free",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=1024
    )
    
    # ✅ STRATEGIC FIX: The Finalizer is now the powerful Llama 3.1 model on Groq to avoid rate limits.
    model_finalizer = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)
    st.success("All AI models initialized successfully!")
except Exception as e:
    st.error(f"Error initializing AI models: {e}", icon="🚨")
    st.info("Please ensure all API keys (Google, Groq, OpenRouter) are correct in your .env file.")
    st.stop()

# --- 3. RAG & DOCUMENT PROCESSING ---
def process_documents(uploaded_files):
    """Loads, splits, and embeds documents into a ChromaDB vector store."""
    all_chunks = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_filepath = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader(temp_filepath) if uploaded_file.name.endswith(".pdf") else Docx2txtLoader(temp_filepath)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)

    if not all_chunks:
        return None

    vector_store = Chroma.from_documents(documents=all_chunks, embedding=embedding_model)
    return vector_store.as_retriever()

# --- 4. MEMORY SETUP ---
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- 5. PROMPT ENGINEERING ---
researcher_prompt = ChatPromptTemplate.from_template(
    """You are an expert AI researcher. Base your answer on the provided context if it is relevant.
    Context: {context}
    Query: {query}"""
)

finalizer_prompt = ChatPromptTemplate.from_template(
    """You are a world-class editor. Review three AI research drafts and synthesize them into a single, superior response, consistent with the conversation history.
    Conversation History: {chat_history}
    Original Query: {query}
    ---
    Draft 1 (Gemini): {draft_gemini}
    ---
    Draft 2 (Llama3 on Groq): {draft_groq}
    ---
    Draft 3 (Mistral on OpenRouter): {draft_openrouter}
    ---
    Produce the final, synthesized response.
    """
)

# --- 6. LANGCHAIN WORKFLOW ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain_gemini = researcher_prompt | model_gemini | StrOutputParser()
chain_groq = researcher_prompt | model_groq_researcher | StrOutputParser()
chain_openrouter = researcher_prompt | model_openrouter | StrOutputParser()

map_chain = RunnableParallel(
    draft_gemini=chain_gemini,
    draft_groq=chain_groq,
    draft_openrouter=chain_openrouter,
)

# --- 7. UI & INTERACTION LOGIC ---
with st.sidebar:
    st.header("📄 Document Analysis")
    st.info("Upload documents to provide the AI team with context. Click 'Process Documents' after uploading.")
    uploaded_files = st.file_uploader("Upload PDF or Word files", type=["pdf", "docx"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents... (This may take a moment on first run as the model downloads)"):
                st.session_state.retriever = process_documents(uploaded_files)
                st.success("Documents processed!")
        else:
            st.warning("Please upload at least one document.")

    st.divider()
    st.header("About")
    st.markdown("This Trio AI Research Assistant combines the strengths of three AI models for comprehensive insights.")
    if st.button("Clear Chat History"):
        st.session_state.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a question to the AI team..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.status("🧠 Consulting the AI research team...", expanded=True) as status:
            context_str = "No documents were provided."
            if 'retriever' in st.session_state:
                status.update(label="Searching documents...")
                retrieved_docs = st.session_state.retriever.invoke(user_prompt)
                context_str = format_docs(retrieved_docs)
            
            researcher_input = {"query": user_prompt, "context": context_str}

            status.update(label="Consulting Gemini...")
            response_gemini = chain_gemini.invoke(researcher_input)
            
            status.update(label="Consulting Llama3 (Groq)...")
            response_groq = chain_groq.invoke(researcher_input)
            
            status.update(label="Consulting Mistral (OpenRouter)...")
            response_openrouter = chain_openrouter.invoke(researcher_input)

            status.update(label="Synthesizing final response...")

        response_placeholder = st.empty()
        full_response = ""
        try:
            finalizer_input = {
                "query": user_prompt, 
                "draft_gemini": response_gemini, 
                "draft_groq": response_groq,
                "draft_openrouter": response_openrouter,
                "chat_history": st.session_state.memory.load_memory_variables({})['chat_history']
            }
            final_chain = finalizer_prompt | model_finalizer | StrOutputParser()
            for chunk in final_chain.stream(finalizer_input):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"An error occurred during final synthesis: {e}"
            response_placeholder.error(full_response, icon="🚨")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.memory.save_context({"input": user_prompt}, {"output": full_response})

