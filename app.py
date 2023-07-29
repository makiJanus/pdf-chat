# Import necessary libraries
import streamlit as st  # Streamlit framework for creating interactive web apps
from ooba_middelware import oobaLLM  # Custom middleware for language model
from PyPDF2 import PdfReader  # Library for working with PDF files
import os  # Library for file and directory operations

# Import custom langchain and related modules
import langchain  # Custom language model framework
from langchain import LLMChain  # Custom language model chain
from langchain import PromptTemplate  # Custom prompt template for language model
from langchain.text_splitter import CharacterTextSplitter  # Text splitter for breaking text into chunks
from langchain.chains.question_answering import load_qa_chain  # Function to load question-answering chain
from langchain.vectorstores import Qdrant  # In-memory vector store for similarity search
from langchain.embeddings import SentenceTransformerEmbeddings  # Sentence embeddings for documents

import streamlit_toggle as tog  # Custom Streamlit widget for toggle switch

import re  # Library for regular expressions
import ast  # Library for safe evaluation of Python literals

# Function to format chat history into a readable string
def format_chat_history(messages):
    chat_string = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        chat_string += f"{role.capitalize()}: {content}\n"

    return chat_string.strip()

# Function for generating language model response
def generate_response(user_question, knowledge_base, chain):
    # Search in the knowledge base for similar documents to the user's question
    docs = knowledge_base.similarity_search(user_question, k=4)
    # Generate a response using the language model chain
    response = chain.run(input_documents=docs, question=user_question)
    # Clean up the response by removing unnecessary tags
    response = re.sub('</s>', '', response)
    return response

# Function to save chat history to a file
def save_chat(messages, title):
    # Generate a file name based on the PDF title or use a default name if the title is not available
    file_name = title if title else "chat_history"

    # Replace any invalid characters in the file name with underscores
    file_name = ''.join(c if c.isalnum() else '_' for c in file_name)

    try:
        # Write chat messages to a text file in UTF-8 encoding
        with open(f"./data/history_chat/{file_name}.txt", "w", encoding="utf-8") as file:
            for message in messages:
                file.write(f"{message}\n")
        
        # Display a success message in Streamlit app
        st.success("Chat history saved successfully!")
    except Exception as e:
        # Display an error message if there's an issue saving the chat history
        st.error(f"Error saving chat history: {e}")


# Function to load chat history from a file
def load_chat(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            messages = []
            for line in lines:
                # Use ast.literal_eval to safely evaluate the dictionary string
                message = ast.literal_eval(line.strip())
                messages.append(message)
            # Set the loaded chat history to the Streamlit session state
            st.session_state.messages = messages
    except Exception as e:
        # You can handle the error here as per your requirement
        print(f"Error loading chat history: {e}")

# Function to get a list of files in a folder
def get_files_in_folder(folder_path):
    try:
        # Get a list of all files in the folder
        files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
        return files
    except Exception as e:
        # You can handle the error here as per your requirement
        print(f"Error getting files from folder: {e}")
        return []

# Disable verbose mode for langchain
langchain.verbose = False

# Set the title of the Streamlit app page
st.set_page_config(page_title="💬 Research Assistant")

# Create a sidebar for additional user input and options
sidebar = st.sidebar

# Sidebar widgets for saving and loading chat history, and uploading PDF files
with sidebar:
    # Header for the sidebar
    st.header("Ask your Researcher 💬")
    
    # Two columns layout for saving chat history and title input
    col1, col2 = st.columns(2)
    
    # Text input for the title of the chat history (default value is "history_title")
    pdf_title = col1.text_input(
        label="",
        value="history_title",
        label_visibility="collapsed",
        placeholder="title.pdf",
        key="placeholder",
    )
    
    # Button to save the chat history, on_click event triggers the save_chat function
    col2.button('save chat history', on_click=lambda: save_chat(messages=st.session_state.messages, title=pdf_title))
    
    # Two columns layout for loading chat history and dropdown selectbox
    col3, col4 = st.columns(2)
    
    # Get a list of files in the folder for loading chat history
    file_options = get_files_in_folder(".\\data\\history_chat")
    file_names = [file.split("\\")[-1][:-4] for file in file_options]
    
    # Dropdown selectbox to choose a previously saved chat history
    option = col3.selectbox(
        label='How would you like to be contacted?',
        label_visibility="collapsed",
        options=file_names
    )
    
    # Button to load the selected chat history, on_click event triggers the load_chat function
    option_path = f".\\data\\history_chat\\{option}.txt"
    col4.button('load chat history', on_click=lambda: load_chat(option_path))
    
    # File uploader widget to allow the user to upload a PDF
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    # Custom toggle switch widget for enabling/disabling PDF-based questions
    switch = tog.st_toggle_switch(
        label="Ask PDF", 
        key="Key1", 
        default_value=False, 
        label_after=True, 
        inactive_color='#D3D3D3', 
        active_color="#11567f", 
        track_color="#29B5E8"
    )

# Import model (oobaLLM)
llm = oobaLLM()

# Template for language model chain prompt
template = """
You're a open minded, curious, and incredibly smart assistant, your ultimate goal is to expand knowledge and its applications to whatever the user wants to explore, only help the user to explore ideas and find knowledge and aplications of it.

{prompt}
"""

# Define a custom prompt template for the language model chain
chain_prompt = PromptTemplate(
    input_variables=["prompt"],
    template=template
)

# Create an instance of the language model chain (llm_chain) with the oobaLLM model and the custom prompt template
llm_chain = LLMChain(llm=oobaLLM(), prompt=chain_prompt)

# Initialize an empty chat history string
chat_history = ""

# Load the question-answering chain using the oobaLLM model
chain = load_qa_chain(llm, chain_type="stuff")

# If the prompt template contains "Helpful Answer:", replace it with "Assistant:" for better presentation
if "Helpful Answer:" in chain.llm_chain.prompt.template:
    chain.llm_chain.prompt.template = (
        f"Human:{chain.llm_chain.prompt.template}".replace("Helpful Answer:", "Assistant:")
    )

# Check if a PDF file is uploaded and process it
if pdf_file:
    # Read the PDF file and extract text from its pages
    pdf = PdfReader(pdf_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    
    # Split the text into chunks for similarity search
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=600, chunk_overlap=100, length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings for text using SentenceTransformer
    embeddings = SentenceTransformerEmbeddings(model_name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6")

    # Create an in-memory Qdrant instance for similarity search
    knowledge_base = Qdrant.from_texts(
        chunks,
        embeddings,
        location=":memory:",
        collection_name="doc_chunks",
    )

# Create or retrieve the chat history list in the Streamlit session state
if "messages" not in st.session_state.keys():
    # If messages list does not exist, create it with a default welcome message from the assistant
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages in the Streamlit app using the Streamlit chat_message container
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Check for user input (user prompts) using Streamlit's chat_input widget
if prompt := st.chat_input(disabled=not True):
    # Append the user's input to the messages list as a new user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        # Display the user's input in the chat interface
        st.write(prompt)
        format_chat_history = format_chat_history(st.session_state.messages)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if switch:
                if pdf_file:
                    # If PDF mode is enabled and a PDF is uploaded, generate a response using the PDF-based approach
                    response = generate_response(prompt, knowledge_base, chain)
                else:
                    # If PDF mode is enabled but no PDF is uploaded, inform the user to upload a PDF
                    response = "If you want to ask a PDF, don't forget to upload it!"
            else:
                # If PDF mode is disabled, generate a response using the language model chain (llm_chain)
                response = llm_chain.run(format_chat_history)
                response = re.sub('</s>', '', response)
                response = re.sub('Assistant:', '', response)
            # Display the response in the chat interface
            st.write(response) 
    # Append the assistant's response to the messages list
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
