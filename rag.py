from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint

import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
warnings.filterwarnings("ignore")
GOOGLE_API_KEY="AIzaSyB2poeL3WYF56p8i0Km96EWIT_6nvFiZ_U"
def load_model():
  #Function to load Model
  model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY,
                             temperature=0.4,convert_system_message_to_human=True)
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
  return model,embeddings
def get_data():
  #Function to load data
  pdf_loader = PyPDFLoader("/content/RAG-USING-GEMINI/2312.10997.pdf.pdf")
  pages = pdf_loader.load_and_split()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
  context = "\n\n".join(str(p.page_content) for p in pages)
  texts = text_splitter.split_text(context)
  model,embeddings=load_model()
  vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
  qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True

)
  return qa_chain
def get_query():
  #Function to get query from the user
  qa_chain=get_data()
  query = "describe about evaluation in RAG"
  output = qa_chain({"query": query})
  print(output["result"])
  get_query()
