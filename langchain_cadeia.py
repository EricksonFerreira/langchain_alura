from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.globals import set_debug
import os
from dotenv import load_dotenv

load_dotenv()
set_debug(True)
numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY")
)

modelo_cidade = ChatPromptTemplate.from_template(
    "Sugira uma cidade dado meu interesse por {interesse}. A sua saída deve ser SOMENTE o nome da cidade. Cidade: "
)

modelo_restaurantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidades}"
)

modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}"
)


cadeia_cidade = LLMChain(llm=llm, prompt=modelo_cidade)
cadeia_restaurantes = LLMChain(llm=llm, prompt=modelo_restaurantes)
cadeia_cultural = LLMChain(llm=llm, prompt=modelo_cultural)

cadeia = SimpleSequentialChain(
    chains=[cadeia_cidade, cadeia_restaurantes, cadeia_cultural],
    verbose=True # log do que está acontecendo
)
 
resultado = cadeia.invoke("praias")
print(resultado)
