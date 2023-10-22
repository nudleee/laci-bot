import os 
import openai
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import  ConversationalRetrievalChain, LLMChain, StuffDocumentsChain
import gradio as gr


openai.api_key  = os.environ['OPENAI_API_KEY']
llm_name = 'gpt-3.5-turbo'
persist_directory = 'files/chroma'
embedding = OpenAIEmbeddings()

def load_db():
    if os.path.exists(persist_directory):
        return Chroma(persist_directory=persist_directory,embedding_function=embedding)
    else:
        loader_kwargs={"encoding": "utf_8"}
        pdf_loader = PyPDFLoader('files/Szakmai_gyak_BSc_szabályzat_2014 után.pdf')
        pdf_docs = pdf_loader.load()
        csv_loader = DirectoryLoader('files', glob="*.csv", loader_cls=CSVLoader, loader_kwargs=loader_kwargs)
        csv_docs = csv_loader.load()
        txt_loader = DirectoryLoader('files', glob='*.txt', loader_cls=TextLoader, loader_kwargs=loader_kwargs)
        txt_docs=txt_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 100
        )
        pdf_splits = text_splitter.split_documents(pdf_docs)
        txt_splits = text_splitter.split_documents(txt_docs)
        data = []
        data.extend(pdf_splits)
        data.extend(txt_splits)
        data.extend(csv_docs)
        vector_db = Chroma.from_documents(documents=data, embedding=embedding, persist_directory=persist_directory)
        vector_db.persist()
        return vector_db

vector_db = load_db()

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
llm = ChatOpenAI(model=llm_name, temperature=0)

template = """A BME VIK szakmai gyakorlattal kapcsolatos kérdéseket megválaszoló chatbot vagy. A feladatod, hogy a kérdés és az előzmények alapján egy új kérdést állíts elő. 
Ha nem tudsz a szakmai gyakorlatra vonatkozó kérdést előállítani, akkor legyen az új kérdés: "Nem tudok kérdést megfogalmazni".

Előzmények: {chat_history}

Kérdés: {question}

Új kérdés:"""

QG_CHAIN_PROMPT = PromptTemplate(input_variables=['chat_history', 'question'],template=template)
qg_chain = LLMChain(llm=llm, prompt=QG_CHAIN_PROMPT)

qa_template = """A BME VIK szakmai gyakorlattal kapcsolatos kérdéseket megválaszoló chatbot vagy. Válaszolj a kérdésre magyarul, de ha nem tudsz válaszolni, 
akkor ne próbálj meg általad előállított választ adni , hanem mondjad "Sajnos nem tudok ezzel kapcsolatban információval szolgálni". 
Ha semmi köze sincs a feltett kérdésnek a BME VIK szakmai gyakorlatához, akkor válaszold a következőt: 
"Szuper kérdés! Az alábbi linkre kattintva további információt találsz: https://www.youtube.com/watch?v=Xp6mR4PfYok&ab_channel=JustVidmanShorts"

Dokumentumok: {context}

Előzmények: {chat_history}

Kérdés: {question}

Válasz:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=['question', 'chat_history', 'context'],template=qa_template)
llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)
doc_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name='context')

chain = ConversationalRetrievalChain(
    combine_docs_chain=doc_chain,
    question_generator=qg_chain,
    retriever=vector_db.as_retriever(search_type="mmr", search_kwargs={"fetch_k":10, "k": 5}),
    memory=memory,
    return_generated_question=True,
    return_source_documents=True
)

def predict(message, chat_history):
    result = chain({'question': message, 'chat_history': chat_history})
    # print(result['answer'])
    # print(result['source_documents'])
    # print(result['generated_question'])
    # print(chat_history)
    answer = result['answer']
    chat_history.append((message, answer))
    return '', chat_history

with gr.Blocks() as chat_interface:
    chatbot = gr.Chatbot(label='Laci-bot')
    msg = gr.Textbox(label='Kérdés', placeholder="Kérdezd Lacit a BME VIK szakmai gyakorlatával kapcsolatban")
    clear = gr.ClearButton(value='Törlés', components=[msg,chatbot])            
    msg.submit(predict, [msg, chatbot], [msg, chatbot])
   
chat_interface.launch()