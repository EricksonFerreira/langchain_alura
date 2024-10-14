from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.globals import set_debug
import os
from dotenv import load_dotenv
from langchain_core.pydantic_v1 import Field,BaseModel
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
set_debug(True)

class Destino(BaseModel):
    cidade = Field("cidade a visitar")
    motivo = Field("motivo da viagem")


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY")
)
parseador = JsonOutputParser(pydantic_object=Destino)

modelo_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado meu interesse por {interesse}. 
    {formatacao_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formatacao_de_saida": parseador.get_format_instructions()}
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
    chains=[
        cadeia_cidade,
             cadeia_restaurantes, 
            cadeia_cultural
    ],
    verbose=True # log do que está acontecendo
)
 
resultado = cadeia.invoke("praias")
print(resultado)
