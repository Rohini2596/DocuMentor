# DocuMentor
An intelligent PDF assistant powered by Ollama's LLM that lets you chat with your documents and get answers from their content.
Features:
  1. PDF Chat: Upload PDFs and ask questions about their content
  2. Conversational Memory: Remembers your chat history per document
  3. Document Management: View, select, and delete uploaded PDFs
  4. General Chat: Ask general questions when no PDF is selected
  5. Local Processing: Runs entirely on your machine (privacy-focused)

Tech Stack:
  Frontend: Streamlit
  LLM: Ollama (llama3.1:8b)
  Embeddings: Nomic-embed-text
  Vector Store: FAISS (local storage)
  LangChain
