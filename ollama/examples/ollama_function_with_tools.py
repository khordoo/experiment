import re
import json
from typing import List, Any
from langchain_experimental.llms.ollama_functions import OllamaFunctions ,convert_to_ollama_tool
from langchain_core.pydantic_v1 import BaseModel,Field
from langchain_core.tools import BaseTool,tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser ,StrOutputParser

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

# print(multiply)
# print(multiply.name)
# print(multiply.description)
# print(multiply.args)


from langchain.tools.render import render_text_description
from langchain.agents import AgentExecutor,create_react_agent

rendered_tools = render_text_description([multiply])
# print(rendered_tools)
# print('conver to tools')
# converted_tools = convert_to_ollama_tool(multiply)
# print(converted_tools)

prompt = PromptTemplate.from_template(
    """
    <|system|>You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys.
<|end|>

<|user|> 
QUESTION: {question}<|end|>
<|assistant|> AI:
 """
)
"""
{'name': '__conversational_response', 'description': 'Respond conversationally if no other tools should be called for a given query.', 'parameters': {'type': 'object', 'properties': {'response': {'type': 'string', 'description': 'Conversational response to the user.'}}, 'required': ['response']}}

"""

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

# model = OllamaFunctions(model="phi3:3.8b-mini-instruct-4k-fp16", format="json", temperature=0)
model = OllamaFunctions(model="llama3:instruct", format="json", temperature=0)
tools = [multiply]
multi_tool_def=[]
function_calls=[]
# print(multiply.name)
# print(multiply.description)
# print(multiply.args)
for t in tools:
    multi_tool_def.append( {
                "name": t.name,
                "description": t.description,
                "parameters": {
                    "type": "object",
                    "properties": t.args
                    },
                "required": [arg for arg in t.args.keys()],
                }
            )
    function_calls.append({'name':t.name})



model= model.bind_tools(
tools=multi_tool_def,
    function_call={'name':multiply.name}
)

# model.bind_tools([{'multiply':rendered_tools}])
chain = prompt | model
res= chain.invoke({"question": "what's thirteen times 4", "rendered_tools":rendered_tools})
print('response:')
print(res)
