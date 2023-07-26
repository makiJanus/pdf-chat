import streamlit as st
from ooba_middelware import oobaLLM
from PyPDF2 import PdfReader

import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings


# Function for generating LLM response
def generate_response(user_question, knowledge_base, chain):
    
    docs = knowledge_base.similarity_search(user_question, k=4)
    # Grab and print response
    response = chain.run(input_documents=docs, question=user_question)
    # response = user_question
    
    return response

#Function so save chat history
def save_chat(messages, title):
    # Generate a file name based on the PDF title or use a default name if the title is not available
    file_name = title if title else "chat_history"
    
    # Replace any invalid characters in the file name with underscores
    file_name = ''.join(c if c.isalnum() else '_' for c in file_name)
    
    try:
        with open(f"./data/history_chat/{file_name}.txt", "w", encoding="utf-8") as file:
            for message in messages:
                file.write(f"{message}\n")
        
        st.success("Chat history saved successfully!")
    except Exception as e:
        st.error(f"Error saving chat history: {e}")


langchain.verbose = False

# App title
st.set_page_config(page_title="💬 PDF Chat")
sidebar= st.sidebar

# PDF Streamlit Uploader
sidebar.header("Ask your PDF 💬")
pdf_file = sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Import model
llm = oobaLLM()
# Load question answering chain
chain = load_qa_chain(llm, chain_type="stuff")

if "Helpful Answer:" in chain.llm_chain.prompt.template:
    chain.llm_chain.prompt.template = (
        f"Human:{chain.llm_chain.prompt.template}".replace(
            "Helpful Answer:", "Assistant:"
        )
    )

if pdf_file:
    # Read PDF file pages and metadata
    pdf = PdfReader(pdf_file)
    pdf_title = str(pdf_file.name).split(".")[0]
    # Collect text from pdf
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
        
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=700, chunk_overlap=100, length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Embedding for text
    embeddings = SentenceTransformerEmbeddings(model_name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6")

    # Create in-memory Qdrant instance
    knowledge_base = Qdrant.from_texts(
        chunks,
        embeddings,
        location=":memory:",
        collection_name="doc_chunks",
    )
    sidebar.button('save chat history', on_click=lambda :save_chat(messages=st.session_state.messages, title=pdf_title))
    
    

#Creating the chatbot interface
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
    
# Display chat messages in a streamlit container
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# User-provided prompt
if prompt := st.chat_input(disabled=not (True)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
                response = generate_response(prompt, knowledge_base, chain) 
                st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
