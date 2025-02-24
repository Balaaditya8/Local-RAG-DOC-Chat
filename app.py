import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama  # Use Ollama for local LLM

# App Title
st.title('📄 RAG-DOC-Chat (Local with Ollama)')

# File Upload
uploaded_file = st.file_uploader("📂 Upload a PDF", type="pdf")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Process PDF (Only once)
if uploaded_file is not None and not st.session_state.pdf_processed:
    with st.spinner("Processing PDF... This may take a few moments."):
        # Save the uploaded file temporarily
        with open('uploaded_file.pdf', "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF and Split into Chunks
        loader = PyPDFLoader('uploaded_file.pdf')
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Convert to Embeddings and Store in ChromaDB
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_documents(
            documents=splits, embedding=embeddings, persist_directory="./chroma_db"
        )

        # Mark PDF as Processed
        st.session_state.pdf_processed = True

    st.success("PDF processing complete! You can now ask questions.")

# Load Ollama with Mistral
if st.session_state.vectorstore and st.session_state.rag_chain is None:
    llm = Ollama(model="mistral")  # Use Mistral model from Ollama

    # Define System Prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use ten sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )

    # Create Chat Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create Document Processing Chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Set up RAG Retrieval Chain
    retriever = st.session_state.vectorstore.as_retriever()
    st.session_state.rag_chain = create_retrieval_chain(retriever, document_chain)

# Display Chat Messages
for input_text, output_text in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(input_text)
    with st.chat_message("assistant"):
        st.write(output_text)

# Capture User Input (Chat Input)
if st.session_state.pdf_processed:
    user_input = st.chat_input("Ask a question about the PDF")
else:
    user_input = None

# Process User Input
if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    # Get response from RAG model
    results = st.session_state.rag_chain.invoke({"input": user_input})
    answer = results["answer"]

    with st.chat_message("assistant"):
        st.write(answer)

    # Append to Chat History
    st.session_state.chat_history.append((user_input, answer))