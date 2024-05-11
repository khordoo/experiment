from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.pydantic_v1 import BaseModel

class AnswerWithJustification(BaseModel):
    '''An answer to the user question along with justification for the answer.'''
    answer: str
    justification: str

llm = OllamaFunctions(model="phi3", format="json", temperature=0)

structured_llm = llm.with_structured_output(AnswerWithJustification,include_raw=True)
res=structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
print('---------------INCLIUDE RAW = TRUE-------------')
print(res)
# -> AnswerWithJustification(
#     answer='They weigh the same',
#     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
# )

structured_llm = llm.with_structured_output(AnswerWithJustification,include_raw=False)
res=structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

print('---------------INCLIUDE RAW = FALSE-------------')
print(res)
print(res.answer)
