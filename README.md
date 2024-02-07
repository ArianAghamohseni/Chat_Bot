# Streamlit + Langchain + LLama.cpp w/ Mistral: Retrieval Augmented Generation

Scrape a website for web content and pdfs and build a conversational ai chatbot from that knowledgebase.

This chatbot has conversational memory and can hold follow up conversations within the same session.

This code was tested on a WordPress Blog and as such has some logic that may not directly  
work on other websites. However, that can be easily fixed by making some tweaks and you will 
need to make some additional tweaks to suit your specific use cases.

Also, while this code can technically run on a computer without a GPU, running it on a GPU is recommended  
since the RAG process can be very slow otherwise.  

You will also need to change how you install `llama-cpp-python` package depending on your OS and whether  
you are planning on using a GPU or not.


Download the `Mistral7b` quantized model from `huggingface` from the following link:
[mistral-7b-instruct-v0.1.Q5_0.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_0.gguf)
