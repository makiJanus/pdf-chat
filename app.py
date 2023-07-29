import streamlit as st
from ooba_middelware import oobaLLM
from PyPDF2 import PdfReader
import os

import langchain
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
import  streamlit_toggle as tog

import re

import ast


def format_chat_history(messages):
    chat_string = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        chat_string += f"{role.capitalize()}: {content}\n"

    return chat_string.strip()

# Function for generating LLM response
def generate_response(user_question, knowledge_base, chain):
    # Search in knowledge_base
    docs = knowledge_base.similarity_search(user_question, k=4)
    # Grab and print response
    response = chain.run(input_documents=docs, question=user_question)
    # response = user_question
    response = re.sub('</s>', '', response)
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


def load_chat(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            messages = []
            for line in lines:
                # Use ast.literal_eval to safely evaluate the dictionary string
                message = ast.literal_eval(line.strip())
                messages.append(message)
            st.session_state.messages = messages
    except Exception as e:
        # You can handle the error here as per your requirement
        print(f"Error loading chat history: {e}")

def get_files_in_folder(folder_path):
    try:
        # Get a list of all files in the folder
        files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
        return files
    except Exception as e:
        # You can handle the error here as per your requirement
        print(f"Error getting files from folder: {e}")
        return []

langchain.verbose = False

# App title
st.set_page_config(page_title="💬 Research assitant")
sidebar= st.sidebar

# Streamlit sidebar
with sidebar:
    st.header("Ask your Researcher 💬")
    col1, col2 = st.columns(2)
    pdf_title = col1.text_input(
        label="",
        value="history_title",
        label_visibility="collapsed",
        placeholder="title.pdf",
        key="placeholder",
    )
    col2.button('save chat history', on_click=lambda :save_chat(messages=st.session_state.messages, title=pdf_title))
    
    col3, col4 = st.columns(2)
    file_options = get_files_in_folder(".\\data\\history_chat")
    file_names = [file.split("\\")[-1][:-4] for file in file_options]
    option = col3.selectbox(
        label='How would you like to be contacted?',
        label_visibility="collapsed",
        options=file_names)
    option_path = f".\\data\\history_chat\\{option}.txt"
    col4.button('load chat history', on_click=lambda : load_chat(option_path))
    
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    switch = tog.st_toggle_switch(
        label="Ask PDF", 
        key="Key1", 
        default_value=False, 
        label_after = True, 
        inactive_color = '#D3D3D3', 
        active_color="#11567f", 
        track_color="#29B5E8"
        )
    
# Import model
llm = oobaLLM()

template = """
You're a open minded, curious, and incredibly smart assistant, your ultimate goal is to expand knowledge and its applications to whatever the user wants to explore, only help the user to explore ideas and find knowledge and aplications of it.

{prompt}
"""

chain_prompt = PromptTemplate(
    input_variables=["prompt"],
    template=template
)

llm_chain = LLMChain( llm=oobaLLM(), prompt=chain_prompt)

llm_chain.run("Hi!")

chat_history = ""

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
    # pdf_title = str(pdf_file.name).split(".")[0]
    # Collect text from pdf
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
        
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=600, chunk_overlap=100, length_function=len
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
        format_chat_history = format_chat_history(st.session_state.messages)
        
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if switch:
                if pdf_file:
                    response = generate_response(prompt, knowledge_base, chain) 
                else:
                    response = "If  you want to ask a pdf, don't forget to upload it!"
            else:
                response = llm_chain.run(format_chat_history)
                response = re.sub('</s>', '', response)
                response = re.sub('Assistant:', '', response)
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)


# f"""
# {Assistant.chat_history}
# """

# I want to develop a female virtual character called Lumina, I want to use her to do erotic cam shows, but I think is important to create a goo sentimental connection with the viewers (male loners mainly). How do you think Lumina can create an emotional dependency from her viewers to her?

# can you make a list of points from that?

