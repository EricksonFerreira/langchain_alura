from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.globals import set_debug
from langchain.memory import ConversationSummaryMemory
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings

load_dotenv()
set_debug(True)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY")
)

carregadores = [
    PyPDFLoader("GTB_standard_Nov23.pdf"),
    PyPDFLoader("GTB_gold_Nov23.pdf"),
    PyPDFLoader("GTB_platinum_Nov23.pdf")
] 
documentos = []
for carregador in carregadores:
    documentos.extend(carregador.load())
 

quebrador = CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
textos = quebrador.split_documents(documentos)
# print(textos)


embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(textos,embeddings)

qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=db.as_retriever())

pergunta = "Como devo proceder caso tenha um item comprado roubado?"

resultado = qa_chain.invoke({"query": pergunta})
print(resultado)

