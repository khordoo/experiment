from langchain_experimental.llms.ollama_functions import OllamaFunctions ,convert_to_ollama_tool
from langchain_core.pydantic_v1 import BaseModel

class AnswerWithJustification(BaseModel):
    '''An answer to the user question along with justification for the answer.'''
    answer: str
    justification: str

dict_schema = convert_to_ollama_tool(AnswerWithJustification)
llm = OllamaFunctions(model="phi3", format="json", temperature=0)
structured_llm = llm.with_structured_output(dict_schema)
res=structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

print('---------------INCLIUDE RAW = FALSE-------------')
print(res)

