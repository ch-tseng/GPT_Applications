import os
from dotenv import load_dotenv
load_dotenv()

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader

#from transformers import AutoModel
import pickle
from pathlib import Path
import os
import streamlit as st
from streamlit_chat import message
import io
import asyncio
import requests
import requests
import html2text
from langchain import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

from langchain.llms import ChatGLM
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from io import StringIO
from urllib.request import urlopen
from bs4 import BeautifulSoup

#DEVICE = "cuda"
#DEVICE_ID = "0"
#CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


st.set_page_config(
    page_title="Discuss an article in depth",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

api_key = os.getenv('OPENAI_API_KEY')
endpoint_url = os.getenv("CHATGPT_API_URL")
doc_archive_path = "docs"
if not os.path.exists(doc_archive_path):
    os.makedirs(doc_archive_path)


async def main():

    async def storeDocEmbeds(file, filename, filetype='pdf'):
        print("Inside the storedoc func")
        if filetype == 'pdf':
            reader = PdfReader(file)
            print("reading content done")
            corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif filetype == 'text':
            stringio = StringIO(file)
            corpus = stringio.read()
        elif filetype == 'txt':
            corpus = file
            print(corpus)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(corpus)

        if option == "ChatGPT":
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        elif  option== "ChatGLM-6B":
            if embedding_model == 'OpenAI(付費)':
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            else:
                embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
                #embeddings = AutoModel.from_pretrained(embedding_model, trust_remote_code=True).cuda()

        vectors = FAISS.from_texts(chunks, embeddings)

        with open(filename, "wb") as f:
            pickle.dump(vectors, f)

    async def getDocEmbeds(file, filename, filetype='pdf', empty=False):
        file_loc = os.path.join(doc_archive_path, filename+".pkl")
        if not os.path.isfile(file_loc) or empty is True:
            await storeDocEmbeds(file, file_loc, filetype)

        with open(file_loc, "rb") as f:
            vectors = pickle.load(f)

        return vectors
    
    async def storeStringEmbeds(input_string, filename):
        global embedding_model

        corpus = input_string

        with open(filename, 'w') as f:
            f.write(input_string)

        loader = TextLoader( filename)
        print("loader: ", loader)
        documents = loader.load()
        print('#'*30)
        print("documents: ", documents)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        if option == "ChatGPT":
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        elif  option== "ChatGLM-6B":
            if embedding_model == 'OpenAI(付費)':
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            else:
                embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
                #embeddings = AutoModel.from_pretrained(embedding_model, trust_remote_code=True).cuda()

        vectors = FAISS.from_documents(chunks, embeddings)

        with open(filename, "wb") as f:
            pickle.dump(vectors, f)

    async def getStringEmbeds(input_string, filename):
        file_loc = os.path.join(doc_archive_path, filename + ".pkl")

        if not os.path.isfile(file_loc):
            await storeStringEmbeds(input_string, file_loc)

        with open(file_loc, "rb") as f:
            vectors = pickle.load(f)

        return vectors



    async def conversational_chat(query):
        result = qa({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    # Creating the chatbot interface
    st.title("Let's discuss an article in depth.")

    option = st.radio("使用的GPT模型: ",("ChatGPT", "ChatGLM-6B"))
    lang_reply = st.radio("Chat回覆的語言: ",("English", "中文"))

    if option == "ChatGPT":
        avatar_style = "fun-emoji"
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    elif  option== "ChatGLM-6B":
        embedding_model = st.selectbox( 'Embedding模型',('paraphrase-multilingual-mpnet-base-v2', 'all-MiniLM-L6-v2', 'multi-qa-MiniLM-L6-cos-v1', 'msmarco-distilbert-base-tas-b' \
                                            , 'all-mpnet-base-v2', 'OpenAI(付費)'))

        if embedding_model in ["paraphrase-multilingual-mpnet-base-v2" ]:
            embedding_model = '../models/' + embedding_model

        avatar_style = "big-smile"
        llm = ChatGLM(
            endpoint_url=endpoint_url,
            max_token=80000,
            top_p=0,
            verbose=True)

    chain = load_qa_chain(llm, chain_type="stuff")

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    #---------------------------------------
    vectors = None
    tSource = st.radio("請選擇資料來源: ",('PDF/TEXT檔', '網址URL', 'copy/paste內容'))
    emptyDOCs = st.checkbox('清除舊的知識，重新讀取。')
    if tSource == 'PDF/TEXT檔':
        uploaded_file = st.file_uploader("Choose a file", type=['pdf','docx','txt'])

        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1]

            with st.spinner("Processing..."):
                if file_type.lower() == 'pdf':
                    uploaded_file.seek(0)
                    file = uploaded_file.read()
                    vectors = await getDocEmbeds(io.BytesIO(file), uploaded_file.name, filetype='pdf', empty=emptyDOCs)
                elif file_type.lower() == 'txt':
                    vectors = await getDocEmbeds(uploaded_file.getvalue().decode("utf-8"), uploaded_file.name, filetype='text', empty=emptyDOCs)

    elif tSource == '網址URL':
        url = st.text_input('URL', '', key='url')
        if url:
            text = ''
            print("URL", url)
            try:
                html = urlopen(url).read()

            except:
                st.write("無法讀取此篇文章!")
                html = None

            if html is not None:
                soup = BeautifulSoup(html, features="html.parser")

                # kill all script and style elements
                for script in soup(["script", "style"]):
                    script.extract()    # rip it out

                # get text
                text = soup.get_text()
                print(text)
                vectors = await getDocEmbeds(text, url.replace('/','').replace('\\',''), filetype='txt', empty=emptyDOCs)

    elif tSource == 'copy/paste內容':
        txtdata = st.text_area('請將文章內容貼於此',"")
        if txtdata:
            text = txtdata.strip()
            vectors = await getDocEmbeds(text, text[:30].replace('/','').replace('\\',''), filetype='txt', empty=emptyDOCs)

    if vectors is not None:
        if option == "ChatGPT":
            qa = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(model_name="gpt-3.5-turbo"),
                retriever=vectors.as_retriever(),
                return_source_documents=True
            )
        elif option == "ChatGLM-6B":
            qa = ConversationalRetrievalChain.from_llm(
                ChatGLM(endpoint_url=endpoint_url,top_p=0),
                retriever=vectors.as_retriever(),
                return_source_documents=True
            )

        st.session_state['ready'] = True


    if st.session_state.get('ready', False):
        if 'generated' not in st.session_state:
            if lang_reply == "English":
                st.session_state['generated'] = ["Welcome! You can now ask any questions about this {}.".format(tSource)]
            else:
                st.session_state['generated'] = ["hi, 您對這{}有什麼問題想問的嗎?".format(tSource)]

        if 'past' not in st.session_state:
            if lang_reply == "English":
                st.session_state['past'] = ["Hey!"]
            else:
                st.session_state['past'] = ["哈囉!"]

        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="e.g: Summarize the document", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                if lang_reply == "English":
                    output = await conversational_chat(user_input + "\n Answer questions in English. Step by step detailed answers.")

                else:
                    output = await conversational_chat(user_input + "\n  用繁體中文回覆問題. 一步一步詳細回答.")

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    if i < len(st.session_state['past']):
                        st.markdown(
                            "<div style='background-color: #90caf9; color: black; padding: 10px; border-radius: 5px; width: 70%; float: right; margin: 5px;'>"+ st.session_state["past"][i] +"</div>",
                            unsafe_allow_html=True
                        )
                    message(st.session_state["generated"][i], key=str(i), avatar_style=avatar_style)

if __name__ == "__main__":
    asyncio.run(main())
