from langchain_community.chat_models import ChatOllama 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate 

llm = ChatOllama(
    model="phi3",
    keep_alive=-1, 
    temperature=0,
    max_new_tokens=521
)

prompt = ChatPromptTemplate.from_template(
    """write me a 500 word article on {topic} 
    from the perspective of a {profession}""" )
chain = prompt | llm | StrOutputParser()

# res = chain.invoke({'topic':'LLM', 'profession':'shipping magnet'})
chunks = []
for s in chain.stream({'topic':'LLM', 'profession':'shipping magnet'}):
        chunks.append(s)
        print(s, end="", flush=True)
# print(res)
# chunks = []
# async def main():
#     async for chunk in llm.astream({'topic':'LLM', 'profession':'shipping magnet'}):
#         chunks.append(chunk)
#         print(chunk.content, end="|", flush=True)
# main() 