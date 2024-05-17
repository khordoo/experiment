import json
import os
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_experimental.llms.ollama_functions import  ChatOllama
from langchain_core.tools import BaseTool,tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.tools import ShellTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_react_agent
os.environ['TAVILY_API_KEY']='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'
shell_tool = ShellTool()
search = TavilySearchResults(max_results=1)
python_repl = PythonREPL()
# You can create the tool to pass to an agent
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`. if you have a big block ",
    func=python_repl.run,
)

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

tools = [pass_through,shell_tool,search,repl_tool]
from langchain.tools.render import render_text_description
from langchain.agents import AgentExecutor,create_react_agent

rendered_tools = [render_text_description([t]) for t in tools]


# converted_tools = convert_to_ollama_tool(multiply)
# print(converted_tools)

# prompt = PromptTemplate.from_template(
#     """
#     <|system|>You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:
#
# {rendered_tools}
# here is the tools name:
# {tool_names}
# {tools}
# Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys.
# <|end|>
#
# <|user|>
# QUESTION: {question}<|end|>
# {agent_scratchpad}
# <|assistant|> AI:
#  """
# )
#TODO: use this for the prompt
prompt = PromptTemplate.from_template(
"""
You are a chatbot that responds to queries by thinking, acting, and observing. In response to a query, first think about the best action, then perform it and observe the result.
{instructions}

<|system|>
TOOLS:

------

You have access to the following tools:

{tools}
Do not pass one tool as input to another tool

To use a tool, please use the following format:

```

Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```

Thought: Do I need to use a tool? No

Final Answer: [your response here]

Example:
For the query 'What is the temperature in the capital of France?':
1. Think: Identify the capital of France using Wikipedia.
2. Act: Fetch capital info using wikipedia: France.
3. Observe: Learn that the capital is Paris.
4. Think: Get Paris's weather.
5. Act: Fetch weather information using weather: Paris.
6. Observe and respond with the temperature in Paris.

You aim to provide accurate, concise answers

```

Begin!

Previous conversation history:
<|end|>
{chat_history}

New input: {input}

{agent_scratchpad}
<|assistant|> AI:
""")
phi3_template= """
You are a chatbot that responds to queries by thinking, acting, and observing. In response to a query, first think about the best action, then perform it and observe the result.
{instructions}

<|system|>
TOOLS:

------

You have access to the following tools:

{tools}
Do not pass one tool as input to another tool

To use a tool, please use the following format:

```

Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```

Thought: Do I need to use a tool? No

Final Answer: [your response here]

Example:
For the query 'What is the temperature in the capital of France?':
1. Think: Identify the capital of France using Wikipedia.
2. Act: Fetch capital info using wikipedia: France.
3. Observe: Learn that the capital is Paris.
4. Think: Get Paris's weather.
5. Act: Fetch weather information using weather: Paris.
6. Observe and respond with the temperature in Paris.

You aim to provide accurate, concise answers

```

Begin!

Previous conversation history:
<|end|>
{chat_history}

New input: {input}

{agent_scratchpad}
<|assistant|> AI:
"""

# model = ChatOllama(model="phi3:3.8b-mini-instruct-4k-fp16", format="json", temperature=0)
model = OllamaFunctions(model="phi3:3.8b-mini-instruct-4k-fp16", format="json", temperature=0)


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



# instructions = """You are an agent designed to write and execute python code to answer questions.
# You have access to a python REPL, which you can use to execute python code.
# If you get an error, debug your code and try again.
# Only use the output of your code to answer the question.
# You might know the answer without running any code, but you should still run the code to get the answer.
# If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
# """
# base_prompt = hub.pull("langchain-ai/react-agent-template")
# prompt = base_prompt.partial(instructions=instructions)
# prompt = prompt.partial(instructions='')





# # Construct the ReAct agent
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

question = "what is current tempeerature in Montreal?"

for step in agent_executor.iter({"input": question}):
    if output := step.get("intermediate_step"):
        action, value = output[0]
        message = json.loads(action.log)
        tool_name = message['Action']
        args= message['Action Input']
        tool_map = {tool.name: tool for tool in tools}
        if tool_name in tool_map:
            chosen_tool = tool_map[tool_name]
            print('------FUNCTION IS CALLED-----:', tool_name)
            print('---ARGS----------------------')
            print(args)
            res= python_repl.run(args)
            # res = chosen_tool.run(args)
            print('RESULT:',res)
        else:

           print('tool ', tool_name , ' not found!')
        # Ask user if they want to continue
        _continue = input("Should the agent continue (Y/n)?:\n") or "Y"
        if _continue.lower() != "y":
            break
# model.bind_tools([{'multiply':rendered_tools}])
# chain = prompt | agent | execute_tool
# res=chain.invoke({"question": "what's twenty plus two hundered?","rendered_tools":rendered_tools})
# print(res)
# res= chain.invoke({"question": "How are you doing today?", "rendered_tools":rendered_tools})
# print(res)
# res= chain.invoke({"question": "show the list of file in current directory?", "rendered_tools":rendered_tools})
# print(res)
# res= chain.invoke({"question": "write a python function that cacuclates a factorial and calculate the factorial of 20", "rendered_tools":rendered_tools ,'agent_scratchpad':[],'tool_names':[],'tools':[]})
# print(res)
