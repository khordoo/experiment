import json
from langchain_community.chat_models import ChatOllama 
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser ,JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.pydantic_v1 import BaseModel, Field

# Pydantic Schema for structured response
class Person(BaseModel):
    name: str = Field(description="The person's name", required=True)
    height: float = Field(description="The person's height", required=True)
    hair_color: str = Field(description="The person's hair color")

context = """Alex is 5 feet tall. 
Claudia is 1 feet taller than Alex and jumps higher than him. 
Claudia is a brunette and Alex is blonde."""

prompt = PromptTemplate.from_template(
    """
    <|user|> {context}
QUESTION: {question}<|end|>
<|assistant|> AI:
 """
)


llm = OllamaFunctions(model="phi3:3.8b-mini-instruct-4k-fp16", 
                      format="json", 
                      temperature=0)






structured_llm = llm.with_structured_output(Person)
chain = prompt | structured_llm

# print(prompt.invoke({'schema':json.dumps(json_schema)}))
# print('scheam:',)
# prompt_str = json.dumps(,indent=2)
# chain = prompt | llm | StrOutputParser()
# chain = prompt | llm | JsonOutputParser() # doess not work as expexted
# res = chain.invoke({'topic':'LLM', 'profession':'shipping magnet'})
# user_description ='a person named John who is 22 years old. Goes to the university every day and expected to be graduatd this year. she loves BBQ and chicken nuggets'
chunks = []

res= chain.invoke({
    "question": "Who is taller?",
    "context": context
    })
print(res)
# chunks = []
# async def main():
#     async for chunk in llm.astream({'topic':'LLM', 'profession':'shipping magnet'}):
#         chunks.append(chunk)
#         print(chunk.content, end="|", flush=True)
# main() 
