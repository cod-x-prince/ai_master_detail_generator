# mentor_ai.py (Upgraded with Resilience)

# --- FIX for Asynchronous Conflict ---
import nest_asyncio
nest_asyncio.apply()
# -----------------------------------

from dotenv import load_dotenv
import os
import tempfile

# Import all necessary model providers and components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.memory import ConversationBufferMemory

# Document Processing Libraries
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
print("✅ .env file loaded.")

# --- 2. MODEL INITIALIZATION ---
try:
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    model_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
    model_groq = ChatGroq(model="llama3-8b-8192", temperature=0.7)
    model_deepseek = ChatTogether(model="deepseek-ai/deepseek-coder-33b-instruct", temperature=0.7)
    model_finalizer = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.4)
    print("✅ All AI models initialized successfully!")
except Exception as e:
    print(f"🚨 Error initializing AI models: {e}")
    print("Please ensure all API keys (Google, Groq, Together) are correct in your .env file.")
    exit()

# --- 3. RAG & DOCUMENT PROCESSING ---
def process_documents_from_folder(folder_path="./documents"):
    """Loads, splits, and embeds documents from a specified folder."""
    if not os.path.exists(folder_path):
        print(f"⚠️  Warning: '{folder_path}' directory not found. No documents will be loaded.")
        return None

    pdf_loader = DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    docx_loader = DirectoryLoader(folder_path, glob="**/*.docx", loader_cls=Docx2txtLoader, show_progress=True)
    
    print("Loading documents...")
    documents = pdf_loader.load() + docx_loader.load()

    if not documents:
        print("No PDF or Word documents found in the folder.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Splitting {len(documents)} documents into {len(chunks)} chunks.")
    print("Creating vector store...")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    print("✅ Vector store created successfully.")
    return vector_store.as_retriever()

# --- 4. MEMORY SETUP ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- 5. PROMPT ENGINEERING ---
researcher_prompt = ChatPromptTemplate.from_template(
    """
    You are an expert AI researcher. Base your answer on the provided context if it is relevant.
    Context from documents: {context}
    User Query: {query}
    """
)

finalizer_prompt = ChatPromptTemplate.from_template(
    """
    You are a world-class editor and synthesizer. Review research drafts and combine them into a single, superior response.
    If a draft indicates a model failed, acknowledge the failure and synthesize the best possible answer from the available drafts.
    Ensure your final answer is consistent with the ongoing conversation history.
    Conversation History: {chat_history}
    Original User Query: {query}
    Draft 1 (Gemini): {draft_gemini}
    Draft 2 (Llama3): {draft_groq}
    Draft 3 (DeepSeek): {draft_deepseek}
    Produce the final, synthesized response.
    """
)

# --- 6. LANGCHAIN WORKFLOW ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain_gemini = researcher_prompt | model_gemini | StrOutputParser()
chain_groq = researcher_prompt | model_groq | StrOutputParser()
chain_deepseek = researcher_prompt | model_deepseek | StrOutputParser()

# --- 7. MAIN EXECUTION LOOP with Error Handling ---
if __name__ == "__main__":
    retriever = process_documents_from_folder()

    print("\n--- 🧠 Trio AI Research Assistant (CLI) ---")
    print("Enter 'exit' to end the conversation.")
    
    while True:
        user_prompt = input("\nYour question: ")
        if user_prompt.lower() == 'exit':
            break

        print("\n🧠 Consulting the AI research team...")
        
        if retriever:
            retrieved_docs = retriever.invoke(user_prompt)
            context_str = format_docs(retrieved_docs)
            print("✅ Context retrieved from documents.")
        else:
            context_str = "No documents were provided."
        
        researcher_input = {"query": user_prompt, "context": context_str}

        # --- Graceful Failure Handling for each researcher ---
        try:
            response_gemini = chain_gemini.invoke(researcher_input)
            print("✅ Gemini draft received.")
        except Exception as e:
            response_gemini = "Model failed to respond due to an error."
            print(f"⚠️ Warning: Gemini model failed. Error: {e}")

        try:
            response_groq = chain_groq.invoke(researcher_input)
            print("✅ Llama3 (Groq) draft received.")
        except Exception as e:
            response_groq = "Model failed to respond due to an error."
            print(f"⚠️ Warning: Llama3 (Groq) model failed. Error: {e}")
            
        try:
            response_deepseek = chain_deepseek.invoke(researcher_input)
            print("✅ DeepSeek draft received.")
        except Exception as e:
            response_deepseek = "Model failed to respond due to an error."
            print(f"⚠️ Warning: DeepSeek model failed. Error: {e}")

        # Synthesize the final response, even if some models failed
        print("Synthesizing final answer...")
        finalizer_input = {
            "query": user_prompt, 
            "draft_gemini": response_gemini, 
            "draft_groq": response_groq,
            "draft_deepseek": response_deepseek,
            "chat_history": memory.load_memory_variables({})['chat_history']
        }
        
        final_chain = finalizer_prompt | model_finalizer | StrOutputParser()
        final_response = final_chain.invoke(finalizer_input)
        
        print("\n--- Final Response ---")
        print(final_response)
        
        memory.save_context({"input": user_prompt}, {"output": final_response})

