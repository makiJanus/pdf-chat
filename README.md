## Chat with your PDF!

![alt text](https://github.com/makiJanus/pdf-chat/blob/main/git_images/screenshot.png?raw=true)

A web-based application to chat with a PDF, I mainly do it to not read papers, just ask the important stuff, the app uses Oobabooga es model loader and a RTX 4090 to load guanaco-7b for the inference, but of course you can test with others models that suits your needs and resources. the main characteristic of this project are:

    - Streamlit Chatbot interface.
    - Langchain to manage the knwoledge creation and context input.
    - Running locally on GPU.
    - Can save history chat.
    - PDF file uploading. 

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
- Guanaco-7B-HF https://huggingface.co/TheBloke/guanaco-7B-HF/tree/main
- Flex-sentence-embbedings https://huggingface.co/flax-sentence-embeddings/all_datasets_v4_MiniLM-L6

## Credits
This is not 100% my code, I just do some modifications and I inted to do some more in the future.
- Original ask pdf with local llama on cpu: https://github.com/wafflecomposite/langchain-ask-pdf-local
- Original webui class (I rename mine as oobaLLM): https://github.com/ChobPT/oobaboogas-webui-langchain_agent
- Incorporation of both with other interface: https://github.com/sebaxzero/LangChain_PDFChat_Oobabooga