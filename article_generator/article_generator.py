import streamlit as st
import os
from langchain.vectorstores import Chroma
import openai
from urllib.request import urlopen
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
import textwrap
from langchain.llms import ChatGLM
from langchain.chains.question_answering import load_qa_chain
from hanziconv import HanziConv

chatglm_api_url = "http://172.30.19.22:8000"
openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = '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'
st.title("文章重製產生器")

def process_text(text, kws):
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in chunks[:3]]

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_texts(chunks, embeddings)
    docs = db.similarity_search(kws)
    
    return docs[0].page_content

def generate_article(text, keyword, writing_style, word_count):
    if tGPT == "ChatGPT":
        llm = OpenAI(temperature=0)

    else:
        #endpoint_url = "http://172.30.19.22:8000"
        llm = ChatGLM(
            endpoint_url=chatglm_api_url,
            max_token=80000,
            top_p=0,
            verbose=True)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
        )
    docs = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in docs[:4]]
    
    chain = load_summarize_chain(llm, 
                             chain_type="map_reduce",
                             verbose = True)

    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(output_summary, width=100)

    if writing_style == "抒情散文":
        a_style = "prose and lyric"
    elif writing_style == "新聞報導":
        a_style = "news report"
    elif writing_style == "研究報告":
        a_style = "research report"
    elif  writing_style== "愛情浪漫故事":
        a_style = "love romance story"
    elif  writing_style== "武俠小說":
        a_style = "martial arts novel"
    elif  writing_style== "心得評論":
        a_style = "suggestions and comments"
    elif writing_style == "詩歌":
        a_style = "poem or lyrics"

    if tArticle == "新文章":
        messages=[
                {"role": "user", "content": "Write a whole new {} about the article below:\n".format(a_style)},
                {"role": "user", "content": wrapped_text},
                {"role": "user", "content": "\nYour new {} focuses on these key points:{}".format(a_style, keyword)},
                {"role": "user", "content": "Your new {} content includes title and body text".format(a_style)},
                {"role": "user", "content": "Your new {} should written in Traditional Chinese, word count is {} words.".format(a_style, word_count)},
        ]

        query =  "Write a whole new {} about article below:\n".format(a_style)
        query += wrapped_text
        query += "\nYour new {} focuses on these key points:{}".format(a_style, keyword)
        query += "Your new {} content includes title and body text".format(a_style)
        query += "Your new {} should written in Traditional Chinese, word count is {} words".format(a_style, word_count)

    elif tArticle == '模仿':
        messages=[
                {"role": "user", "content": "Write an {} mimic the writing style of the article below:\n".format(a_style)},
                {"role": "user", "content": wrapped_text},
                {"role": "user", "content": "Your {} should focuses on these key points:{}".format(a_style, keyword)},
                {"role": "user", "content": "Your {} contains includes title and body text".format(a_style)},
                {"role": "user", "content": "Your {} should written in Traditional Chinese, word count is {}".format(a_style, word_count)},
        ]

        query = "Write an {} mimic the writing style of the article below:\n".format(a_style)
        query += wrapped_text
        query += "\nYour {} should focuses on these key points:{}\n".format(a_style, keyword)
        query += "Your {} contains includes title and body text.".format(a_style)
        query += "Your {} should written in Traditional Chinese, word count is {} words.".format(a_style, word_count)


    else:
        messages=[
                {"role": "user", "content": "Write an {} about your thoughts on reading the article below:\n".format(a_style)},
                {"role": "user", "content": wrapped_text},
                {"role": "user", "content": "Your thoughts should focuses on these key points:{}".format(keyword)},
                {"role": "user", "content": "Your {} contains includes title and body text".format(a_style)},
                {"role": "user", "content": "Your {} should written in Traditional Chinese, word count is {}".format(a_style, word_count)},
        ]

        query = "Write an {} about your thoughts on reading the article below:\n".format(a_style)
        query += wrapped_text
        query += "\nYour thoughts should focuses on these key points:{}\n".format(keyword)
        query += "Your {} contains includes title and body text.".format(a_style)
        query += "Your {} should written in Traditional Chinese, word count is {} words.".format(a_style, word_count)


    if tGPT == "ChatGPT":
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=3000,
            messages = messages
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content

    else:
        result = llm(query)

    tc = HanziConv.toTraditional(result)
    tc = tc.replace('齣','出').replace('傢','家').replace('家俱','傢俱').replace('颱','台').replace('麵','面')
    tc = tc.replace('泡面','泡麵').replace('面條','麵條').replace('睏','困').replace('迴','回').replace('回響','迴響')
    tc = tc.replace('瞭','了').replace('了解','瞭解').replace('明了','明瞭')
    tc = tc.replace('嚮','向').replace('音向','音響').replace('向應','嚮應').replace('向導','嚮導')
    tc = tc.replace('纔','才').replace('錶','表').replace('手表','手錶').replace('表帶','錶帶').replace('鐘表','鐘錶')
    tc = tc.replace('隻','只')
    result = tc.replace('回盪','迴盪').replace('回廊','迴廊').replace('閤','合').replace('合家','閤家')

    return result

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

genDoc = False
tArticle = st.radio("文章類型: ", ('新文章', '模仿', '讀後感想'))
tLangResponse = st.radio("請選擇文章的語言: ",('中文', '英文'))
keyword = st.text_input("加入文章元素: ")
writing_style = st.selectbox("文章型態:", ["新聞報導", "研究報告", "抒情散文", "愛情浪漫故事", "武俠小說", "心得評論", "詩歌"])
word_count = st.slider("Select word count:", min_value=300, max_value=1000, step=100, value=300)
tGPT = st.radio("使用的GPT模型: ",("ChatGPT", "ChatGLM-6B"))

tSource = st.radio("請選擇參考的文章來源: ",('網址URL', 'PDF/TEXT檔', '文字框輸入'))
if tSource == "網址URL":
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

            for script in soup(["script", "style"]):
                script.extract()    # rip it out

            text = soup.get_text()
            genDoc = True
    
elif tSource == "PDF/TEXT檔":
    pdf = st.file_uploader('Upload your TXT/PDF Document', type=['pdf','txt'])
    if pdf:
        file_type = pdf.name.split('.')[-1]
        text = ''
        if file_type.lower() == 'pdf':
            pdf_reader = PdfReader(pdf)
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file_type.lower() == 'txt':
            stringio = StringIO(pdf.getvalue().decode("utf-8"))
            text = stringio.read()
            
        genDoc = True
        
elif tSource == "文字框輸入":
    text = st.text_area('請將資料內容貼於此',"")
    if text: genDoc = True

if genDoc is True:
    submit_button = st.button("Generate Article")

    if submit_button:
        message = st.empty()
        message.text("Busy generating...")
        
        article = generate_article(text, keyword, writing_style, word_count)
        print('Article', article)
        message.text("")
        st.write(article)
        st.download_button(
            label="Download article",
            data=article,
            file_name= 'Article.txt',
            mime='text/txt',
        )
