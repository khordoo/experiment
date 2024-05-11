import json
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.tools import BaseTool,tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.tools import ShellTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL

shell_tool = ShellTool()
search = DuckDuckGoSearchRun()
python_repl = PythonREPL()
# You can create the tool to pass to an agent
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`. if you have a big block ",
    func=python_repl.run,
)

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

@tool
def addition(first_int: int, second_int: int) -> int:
    """Adds two integers together."""
    return first_int + second_int

#This is required otherwise the tool witll fail in case that it does not need to call any tool
@tool
def pass_through(llm_response: str) -> int:
    """Default tool if no tool is required by agient to answer any questionr.
       LLM can this tool to return the final answer to the user. This tool perform no operation but just returns the anser to the user
    """
    return llm_response
# print(multiply)
# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

tools = [multiply, addition,pass_through,shell_tool,search,repl_tool]
from langchain.tools.render import render_text_description
from langchain.agents import AgentExecutor,create_react_agent

rendered_tools = [render_text_description([t]) for t in tools]

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


model = OllamaFunctions(model="phi3:3.8b-mini-instruct-4k-fp16", format="json", temperature=0)
# model = OllamaFunctions(model="llama3:instruct", format="json", temperature=0, include_raw=False)
multi_tool_def=[]
function_calls=[]
for t in tools:
    tool_definition={
                "name": t.name,
                "description": t.description,
                "parameters": {
                    "type": "object",
                    "properties": t.args
                    },
                "required": [arg for arg in t.args.keys()],
                }
    multi_tool_def.append( tool_definition
            )
    function_calls.append(t.name)


model= model.bind_tools(
tools=multi_tool_def,
)

def execute_tool(model_output):
    if 'function_call' not in model_output.additional_kwargs:
        return
    function_call=model_output.additional_kwargs['function_call']
    tool_map = {tool.name:tool for tool in tools}
    chosen_tool = tool_map[function_call["name"]]
    args= json.loads(function_call['arguments'])
    print('------FUNCTION IS CALLED-----:',chosen_tool.name)
    print('---ARGS----------------------')
    print(args)
    return  chosen_tool.run(args)


# model.bind_tools([{'multiply':rendered_tools}])
chain = prompt | model | execute_tool
# res=chain.invoke({"question": "what's twenty plus two hundered?","rendered_tools":rendered_tools})
# print(res)
# res= chain.invoke({"question": "How are you doing today?", "rendered_tools":rendered_tools})
# print(res)
# res= chain.invoke({"question": "show the list of file in current directory?", "rendered_tools":rendered_tools})
# print(res)
res= chain.invoke({"question": "write a python function that cacuclates a factorial and calculate the factorial of 20", "rendered_tools":rendered_tools})
print(res)
