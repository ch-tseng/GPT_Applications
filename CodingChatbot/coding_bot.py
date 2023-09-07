import streamlit as st
import os, json
from dotenv import load_dotenv
load_dotenv()

import requests

from langchain.llms import ChatGLM

# from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
api_key = os.getenv('OPENAI_API_KEY')


import streamlit as st
from streamlit_chat import message
st.set_page_config(page_title="I'm a coding-bot, help you on coding.", page_icon=":robot:")

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
st.header("LangChain Demo")
option = st.radio("使用的GPT模型: ",("Code-Llama-7B", "ChatGPT-4"	))

"""Python file to serve as the frontend"""
from streamlit_chat import message

from itertools import zip_longest
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

global bot_history,user_history
chat_gpt = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=api_key)
chatglm = ChatGLM(endpoint_url=os.getenv('CHATGPT_API_URL'),top_p=0)


def get_response(history,user_message,temperature=0):

    if option == "ChatGPT-4":
        DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is teaching the user how to program.
        It folows the previous conversation to do so

        Relevant pieces of previous conversation:
        {context}


        (You do not need to use these pieces of information if not relevant)


        Current conversation:
        Human: {input}
        AI:"""

        PROMPT = PromptTemplate(
            input_variables=['context','input'], template=DEFAULT_TEMPLATE
        )

        
        conversation_with_summary = LLMChain(
            llm=chat_gpt,
            prompt=PROMPT,
            verbose=False
        )
        
        response =conversation_with_summary.predict(context=history,input=user_message)
        
    else:

        headers = {
            'Content-Type': 'application/json',
        }

        data = '{"prompt": "'+user_message+'", "history": [], "max_length":"1024" }'
        data = data.encode('utf-8')

        r = requests.post('http://127.0.0.1:8000/', headers=headers, data=data).text
        
        if True:
            json_r = json.loads(r)
            txt_re = json_r["response"]
            print("[Before:]---->", txt_re)
            txt_re = txt_re.replace('\\begin{code}', '```python')
            txt_re = txt_re.replace('\\end{code}', '```')
            print("[After:]--->", txt_re)
            response = {"response":txt_re}
            
        #except:
        #    response = {"response":""}
        
                
    return response

def get_history(history_list):
    history = 'input: I want you to act as an chatbot that teaches coding. The user will mention the topic and language in the specific language he wants to learn. your job is to create a small interactive tutorials in which you break a problem into small coding tasks. Ask the user to complete each task, evaluate its code and then give the next task \n'
    for message in history_list:
        if message['role']=='user':
            history=history+'input '+message['content']+'\n'
        elif message['role']=='assistant':
            history=history+'output '+message['content']+'\n'
    
    return history



# chain = load_chain()

# From here down is all the StreamLit UI.


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_area("You: ", "", key="input")
    return input_text


user_input = get_text()

if user_input:
    user_history = list(st.session_state["past"])
    bot_history = list(st.session_state["generated"])


    combined_history = []

    for user_msg, bot_msg in zip_longest(user_history, bot_history):
        if user_msg is not None:
            combined_history.append({'role': 'user', 'content': user_msg})
        if bot_msg is not None:
            combined_history.append({'role': 'assistant', 'content': bot_msg})

    formatted_history = get_history(combined_history)
    print('formatted_history', formatted_history)

    output = get_response(formatted_history,user_input)
    
    if option == "Code-Llama-7B":
        output = output["response"]

    st.session_state.past.append(user_input)
    st.session_state.generated.append( output )

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])-1,-1, -1):
        print('i', i, st.session_state["past"][i])
        print('i', i, st.session_state["generated"][i])
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

        message(st.session_state["generated"][i], key=str(i))

