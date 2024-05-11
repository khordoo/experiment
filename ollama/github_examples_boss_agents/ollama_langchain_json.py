import json
from langchain_community.chat_models import ChatOllama 
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser ,JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.pydantic_v1 import BaseModel, Field

class Person(BaseModel):
    name: str = Field(..., title="Name", description="The person's name")
    age: int = Field(..., title="Age", description="The person's age")
    job: str = Field(..., title="Name", description="The person's job")
    favorite_food: str = Field(..., title="Fav Food", description="The person's favorite food")


json_schema = {
    "occupation": "Person's job",
    "description": "general information about a person.",
    "type": "object",
    "properties": {
        "name": {"title": "Name", "description": "The person's name", "type": "string"},
        "age": {"title": "Age", "description": "The person's age", "type": "integer"},
        "favorite_food": {
            "title": "Fav Food",
            "description": "The person's favorite food",
            "type": "string",
        },
    },
    "required": ["name", "age","fav_food"],
}


llm = ChatOllama(
    model="phi3:3.8b-mini-instruct-4k-fp16",
    format='json',
    keep_alive=-1, 
    temperature=0,
    max_new_tokens=521
)

# LM Studio
# llm: ChatOpenAI = ChatOpenAI(
#     base_url= "http://localhost:1234/v1",
#     temperature=0,
#     api_key="not-needed"
# )




# llm: ChatOpenAI = ChatOpenAI(
#     temperature=0,
# )


prompt = ChatPromptTemplate.from_template(
        """
Please tell me about a person using the following JSON schema:
{schema}
Now, considering the previous schema, tell me about the followign person: 
{description}
Only output the joson. Do not provide any extra comments.
Example: 
a person named Kate who is 35 years old. she works in a hospital and and loves pizza and Jazz music
Expected Output from AI:
{{ "name": "Kate",
    "age": 35,
    "job": "Hospital worker",
    "favorite_food": "Pizza"
}}
"""
)
# messages = [
#     HumanMessage(
#         content="Please tell me about a person using the following JSON schema:"
#     ),
#     HumanMessage(content="{schema}"),
#     HumanMessage(
#         content="Now, considering the schema, tell me about a a person named Kate who is 35 years old. she works in a hospital and and loves pizza and Jazz music"
#     ),
# ]



# print(prompt.invoke({'schema':json.dumps(json_schema)}))
# print('scheam:',)
# prompt_str = json.dumps(,indent=2)
chain = prompt | llm | StrOutputParser()
# chain = prompt | llm | JsonOutputParser() # doess not work as expexted
# res = chain.invoke({'topic':'LLM', 'profession':'shipping magnet'})
user_description ='a person named John who is 22 years old. Goes to the university every day and expected to be graduatd this year. she loves BBQ and chicken nuggets'
chunks = []

for s in chain.stream({'schema':json.dumps(json_schema,indent=2),  'description':user_description}):
        chunks.append(s)
        print(s, end="", flush=True)
# print(res)
# chunks = []
# async def main():
#     async for chunk in llm.astream({'topic':'LLM', 'profession':'shipping magnet'}):
#         chunks.append(chunk)
#         print(chunk.content, end="|", flush=True)
# main() 