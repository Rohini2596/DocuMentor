import streamlit as st
import os
import json
import shutil
import time
from datetime import datetime
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
st.set_page_config(page_title="DocuMentor", page_icon="🤖", layout="wide")
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
llm = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")
prompt_template = """
You are an assistant for answering questions based on provided PDF content and prior conversation history.
Use the following context from the PDF and the conversation so far to answer the current question.
Conversation history:
{chat_history}
PDF Context:
{context}
Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
def get_history_path(db_name):
    """Get path for chat history JSON file"""
    return "general_chat.json" if db_name == "general_chat" else f"{db_name}_chat.json"
def save_chat_history(db_name, history):
    """Save chat history to JSON file (now handles general chats too)"""
    try:
        with open(get_history_path(db_name), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")
def load_chat_history(db_name):
    """Load chat history from JSON file (supports both formats)"""
    path = get_history_path(db_name)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [{
                    "question": item[0] if isinstance(item, (list, tuple)) else item.get("question", ""),
                    "answer": item[1] if isinstance(item, (list, tuple)) else item.get("answer", ""),
                    "timestamp": datetime.now().isoformat() if isinstance(item, (list, tuple)) else item.get("timestamp")
                } for item in data]
        except Exception as e:
            st.error(f"Error loading chat history: {str(e)}")
            return []
    return []
def format_docs(docs):
    """Format documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)
def format_chat_history(history):
    """Format chat history for prompt"""
    return "\n".join(f"User: {item['question']}\nAssistant: {item['answer']}" for item in history)
def delete_pdf_db(db_name):
    """Delete PDF database and associated files"""
    try:
        if os.path.exists(db_name):
            shutil.rmtree(db_name, ignore_errors=True)
        history_path = get_history_path(db_name)
        if os.path.exists(history_path):
            os.remove(history_path)
        time.sleep(0.3)
        return True
    except Exception as e:
        st.error(f"Error deleting files: {str(e)}")
        return False
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {
        "general_chat": load_chat_history("general_chat")
    }
if "current_db" not in st.session_state:
    st.session_state.current_db = None
if "last_selected_pdf" not in st.session_state:
    st.session_state.last_selected_pdf = None
if "show_delete_confirm" not in st.session_state:
    st.session_state.show_delete_confirm = False
if "need_clear_question" not in st.session_state:
    st.session_state.need_clear_question = False
with st.sidebar:
    st.header("📂DocuMentor File Panel")
    db_dirs = [f for f in os.listdir() if f.startswith("db_") and os.path.isdir(f)]
    available_pdfs = [f.replace("db_", "") for f in db_dirs]
    selected_pdf = st.selectbox("Select a PDF", [""] + available_pdfs, index=0)
    if st.session_state.last_selected_pdf is not None and selected_pdf == "":
        st.session_state.need_clear_question = True
        st.session_state.current_db = None
    if selected_pdf:
        DB_NAME = f"db_{selected_pdf}"
        st.session_state.current_db = DB_NAME
        if DB_NAME not in st.session_state.chat_histories:
            st.session_state.chat_histories[DB_NAME] = load_chat_history(DB_NAME)
        if st.button(f"Delete {selected_pdf}"):
            st.session_state.show_delete_confirm = True
        if st.session_state.show_delete_confirm:
            st.warning(f"Are you sure you want to permanently delete '{selected_pdf}'?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, delete it"):
                    if delete_pdf_db(DB_NAME):
                        del st.session_state.chat_histories[DB_NAME]
                        if st.session_state.current_db == DB_NAME:
                            st.session_state.current_db = None
                            st.session_state.last_selected_pdf = None
                        st.session_state.show_delete_confirm = False
                        st.rerun()
            with col2:
                if st.button("No, keep it"):
                    st.session_state.show_delete_confirm = False
                    st.rerun()
    uploaded_file = st.file_uploader("Upload New PDF", type=["pdf"])
    if uploaded_file:
        new_pdf_name = os.path.splitext(uploaded_file.name)[0]
        DB_NAME = f"db_{new_pdf_name}"
        st.session_state.current_db = DB_NAME
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name       
        with st.spinner("Processing PDF..."):
            try:
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = splitter.split_documents(documents)
                if os.path.exists(DB_NAME):
                    vectorstore = FAISS.load_local(DB_NAME, embeddings, allow_dangerous_deserialization=True)
                    vectorstore.add_documents(docs)
                else:
                    vectorstore = FAISS.from_documents(docs, embeddings)
                vectorstore.save_local(DB_NAME)
                os.unlink(tmp_path)
                st.success(f"{uploaded_file.name} processed and saved!")
                st.session_state.chat_histories[DB_NAME] = []
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                os.unlink(tmp_path)
    if st.session_state.current_db:
        current_history = st.session_state.chat_histories.get(st.session_state.current_db, [])
        if current_history:
            st.markdown(f"### {st.session_state.current_db.replace('db_', '')} History")
            if st.button("🧹 Clear This History"):
                st.session_state.chat_histories[st.session_state.current_db] = []
                save_chat_history(st.session_state.current_db, [])
                st.rerun()
            for i, item in enumerate(current_history[-5:]):
                with st.expander(f"Q{i+1}: {item['question'][:50]}...", expanded=False):
                    st.markdown(f"**You:** {item['question']}")
                    st.markdown(f"**AI:** {item['answer']}")
                    st.caption(f"{datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("No chat history for this document yet.")
    else:
        general_history = st.session_state.chat_histories.get("general_chat", [])
        if general_history:
            st.markdown("### General Chat History")
            if st.button("🧹 Clear General History"):
                st.session_state.chat_histories["general_chat"] = []
                save_chat_history("general_chat", [])
                st.rerun()
            for i, item in enumerate(general_history[-3:]):
                with st.expander(f"General Q{i+1}: {item['question'][:30]}...", expanded=False):
                    st.markdown(f"**You:** {item['question']}")
                    st.markdown(f"**AI:** {item['answer']}")
                    st.caption(f"{datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("No general chat history yet.")
st.title("🤖 DocuMentor")
if st.session_state.need_clear_question:
    if not st.session_state.get("question"):
        st.session_state.question = ""
        st.session_state.last_selected_pdf = None
        st.session_state.need_clear_question = False
        st.rerun()
if st.session_state.last_selected_pdf != st.session_state.current_db:
    st.session_state.last_selected_pdf = st.session_state.current_db
placeholder_text = "Ask any question from the selected PDF..." if st.session_state.current_db else "Ask me anything (general chat)..."
question = st.text_input(
    "🔎 Ask a question:",
    value=st.session_state.get("question", ""),
    key="question",
    placeholder=placeholder_text
)
if question:
    with st.spinner("Thinking..."):
        try:
            db_name = st.session_state.current_db or "general_chat"
            chat_history = st.session_state.chat_histories.get(db_name, [])
            if db_name != "general_chat" and os.path.exists(db_name):
                vectorstore = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                docs = retriever.invoke(question)
                context = format_docs(docs)
                history_str = format_chat_history(chat_history)
                full_input = prompt.format(question=question, context=context, chat_history=history_str)
            else:
                full_input = question
            response = llm.invoke(full_input)
            parsed_response = StrOutputParser().invoke(response)
            
            new_entry = {
                "question": question,
                "answer": parsed_response,
                "timestamp": datetime.now().isoformat()
            }
            chat_history.append(new_entry)
            st.session_state.chat_histories[db_name] = chat_history
            save_chat_history(db_name, chat_history)
            st.markdown("### Answer")
            st.success(parsed_response)
            st.balloons()
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
