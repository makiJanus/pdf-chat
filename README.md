## Chat with your PDF!

![alt text](https://github.com/makiJanus/pdf-chat/blob/main/git_images/screenshot.png?raw=true)

A web-based application for chatting with a PDF. The primary purpose of this application is to avoid reading lengthy papers by allowing users to inquire about essential information. Additionally, you have the option to modify responses if they find them biased due to information retrieval from the PDF. Also it is possible to engage in conversations with the Language Model (LLM) about ideas using just the history chat.

The application utilizes Oobabooga's model loader and an RTX 4090 to load the WizardLM-13B for inference. However, you can experiment with other models that better suit their needs and available resources. The main characteristics of this project include:

    - Streamlit Chatbot interface.
    - Langchain to manage the knwoledge creation and context input.
    - Running locally on GPU.
    - Can save history chat.
    - Can load history chat.
    - PDF file uploading. 
    - It can toggle it answer for pdf based knowledge or LLM model general knowledge.

## To install
1.- Install Oobabooga: https://github.com/oobabooga/text-generation-webui

2.- Clone this git in your computer.

## To run
1.- Start Oobabooga
```
cd text-generation-webui
python server.py --auto-devices --model-menu --api
```
2.- Open start_windows.bat
3.- Run this git in your python environment.
```
cd pdf-chat
streamlit run app.py
```

## Model and Embeddings used for testing
- WizardLM-13B-V1.0-Uncensored-GPTQ https://huggingface.co/TheBloke/WizardLM-13B-V1.0-Uncensored-GPTQ
- Flex-sentence-embbedings https://huggingface.co/flax-sentence-embeddings/all_datasets_v4_MiniLM-L6

## Credits
This is not 100% my code, I just do some modifications and I inted to do some more in the future.
- Original ask pdf with local llama on cpu: https://github.com/wafflecomposite/langchain-ask-pdf-local
- Original webui class (I rename mine as oobaLLM): https://github.com/ChobPT/oobaboogas-webui-langchain_agent
- Incorporation of both with other interface: https://github.com/sebaxzero/LangChain_PDFChat_Oobabooga