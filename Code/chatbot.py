import streamlit as st
from bs4 import BeautifulSoup
import io
import fitz
import requests
from langchain.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


@st.cache_data
def get_page_urls(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    links = [link['href'] for link in soup.find_all('a') if link['href'].startswith(url) and link['href'] not in [url]]
    links.append(url)
    return set(links)


def get_url_content(url):
    response = requests.get(url)
    if url.endswith('.pdf'):
        pdf = io.BytesIO(response.content)
        file = open('pdf.pdf', 'wb')
        file.write(pdf.read())
        file.close()
        doc = fitz.open('pdf.pdf')
        return (url, ''.join([text for page in doc for text in page.get_text()]))
    else:
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.find_all('div', class_='wpb_content_element')
        text = [c.get_text().strip() for c in content if c.get_text().strip() != '']
        text = [line for item in text for line in item.split('\n') if line.strip() != '']
        arts_on = text.index('ARTS ON:')
        return (url, '\n'.join(text[:arts_on]))


@st.cache_resource
def get_retriever(urls):
    all_content = [get_url_content(url) for url in urls]
    documents = [Document(page_content=doc, metadata={'url': url}) for (url, doc) in all_content]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})
    return retriever


@st.cache_resource
def create_chain(_retriever):
    n_gpu_layers = 40
    n_batch = 2048

    llm = LlamaCpp(
            model_path="Model/mistral-7b-instruct-v0.1.Q5_0.gguf",
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=2048,
            temperature=0,
            verbose=False,
            streaming=True,
            )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=_retriever, memory=memory, verbose=False
    )

    return qa_chain


st.set_page_config(
    page_title="Welcome"
)
st.header("Your own ChatBot!")

if "base_url" not in st.session_state:
    st.session_state.base_url = ""

base_url = st.text_input("Enter the site url here", key="base_url")

if st.session_state.base_url != "":
    urls = get_page_urls(base_url)

    retriever = get_retriever(urls)
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I help you today?"}
        ]

    if "current_response" not in st.session_state:
        st.session_state.current_response = ""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    llm_chain = create_chain(retriever)

    if user_prompt := st.chat_input("Your message here", key="user_input"):

        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
        )

        with st.chat_message("user"):
            st.markdown(user_prompt)

        response = llm_chain.run(user_prompt)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        with st.chat_message("assistant"):
            st.markdown(response)
