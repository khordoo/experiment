import json
from langchain_community.chat_models import ChatOllama 
from langchain_experimental.llms.ollama_functions import OllamaFunctions, convert_to_ollama_tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser ,JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional, Type

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import tool ,BaseTool
from langchain.agents import AgentExecutor
@tool("get_current_weather") 
def get_current_weather(locaton:str,):
    """used to get current weather for the specified location"""
    import random 
    return random.randint(-10,40)
class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")

class CustomCalculatorTool(BaseModel):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(
        self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return a * b

    async def _arun(
        self,
        a: int,
        b: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Calculator does not support async")

llm = OllamaFunctions(model="phi3:3.8b-mini-instruct-4k-fp16", 
                      format="json", 
                      temperature=0)



tools = [CustomCalculatorTool()]
tools = [convert_to_ollama_tool(t) for t in tools]
# llm= llm.bind_tools(
#     tools=[
#         {
#             "name": "get_current_weather",
#             "description": "Get the current weather in a given location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {
#                         "type": "string",
#                         "description": "The city and state, " "e.g. San Francisco, CA",
#                     }
#                 },
#                 "required": ["location"],
#             },
#         }
#     ],
#     function_call={"name": "get_current_weather"},
# )
llm= llm.bind_tools(
    tools=tools
)
res= llm.invoke("what is 234 * 345?")
print(res)



# res= llm.invoke("what is the weather in Boston?")
# print(res)
# agent_exexutor = AgentExecutor(agent=llm,tools=tools,verbose=True)
# res = agent_exexutor.invoke({"input":"what is the temperature is Monteal?"})
# print(res)
#print(get_current_weather.run(tool_input='london'))

# print(res)
# multiply = CustomCalculatorTool()
# print(multiply.name)
# print(multiply.description)
# print(multiply.args)
# print(multiply.return_direct)