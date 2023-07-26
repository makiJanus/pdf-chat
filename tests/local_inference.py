from ooba_middelware import oobaLLM

import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings

langchain.verbose = False

# Test libs
from langchain.document_loaders import PyPDFLoader
#


llm = oobaLLM()

# Load question answering chain
chain = load_qa_chain(llm, chain_type="stuff")

if "Helpful Answer:" in chain.llm_chain.prompt.template:
    chain.llm_chain.prompt.template = (
        f"### Human:{chain.llm_chain.prompt.template}".replace(
            "Helpful Answer:", "\n### Assistant:"
        )
    )

# PDF Loader
loader = PyPDFLoader("./paper1.pdf")
pdf = loader.load_and_split()
#

if pdf:
    # pdf_reader = PdfReader(pdf)

    # Collect text from pdf
    text = ""
    for page in pdf:
        text += page.page_content

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=700, chunk_overlap=100, length_function=len
    )
    chunks = text_splitter.split_text(text)

    #embeddings = SentenceTransformerEmbeddings(model_name='hku-nlp/instructor-large')
    embeddings = SentenceTransformerEmbeddings(model_name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6")

    # Create in-memory Qdrant instance
    knowledge_base = Qdrant.from_texts(
        chunks,
        embeddings,
        location=":memory:",
        collection_name="doc_chunks",
    )

    # user_question = st.text_input("Ask a question about your PDF:")
user_question = "What are the keypoints in Emily research?"

if user_question:
    docs = knowledge_base.similarity_search(user_question, k=4)

    # Calculating prompt (takes time and can optionally be removed)
    prompt_len = chain.prompt_length(docs=docs, question=user_question)
    # st.write(f"Prompt len: {prompt_len}")
    # Grab and print response
    response = chain.run(input_documents=docs, question=user_question)
    # st.write(response)
    print(response)