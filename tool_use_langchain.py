# %%
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import CohereEmbeddings
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import sys
import uuid
from psycopg_pool import ConnectionPool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableLambda, RunnableConfig, Runnable
from IPython.display import Image, display
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# %%
# Set API keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Optionally, add tracing
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_ENDPOINT"] = 'https://api.smith.langchain.com'
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

os.environ["LANGCHAIN_PROJECT"] = "llama3-tool-use-agent"

# %%
# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# %%
@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2

@tool
def web_search(input: str) -> str:
    """Runs web search."""
    web_search_tool = TavilySearchResults()
    docs = web_search_tool.invoke({"query": input})
    return docs

@tool
def image2text(image_url: str, prompt: str) -> str:
    """generate text for image_url based on prompt."""
    input = {
        "image": image_url,
        "prompt": prompt
    }
    output = replicate.run(
        "yorickvp/llava-13b:b5f612031823083fd4b6dda3e32fd8a0e75dc39d8a4191bb742157fb",
        input=input
    )
    return "".join(output)

@tool
def text2speech(text: str) -> int:
    """convert text to a speech."""
    output = replicate.run(
        "cjwbw/seamless_communication:668a4fec05a8871a54e5fe8d45df25ec4c794dd43169b9a11562309b2d45873b0",
        input={
            "task_name": "T2ST (Text to Speech translation)",
            "input_text": text,
            "input_text_language": "English",
            "max_input_audio_length": 60,
            "target_language_text_only": "English",
            "target_language_with_speech": "English"
        }
    )
    return output

tools = [magic_function, web_search, image2text, text2speech]

print(magic_function)
magic_function.args_schema

# %%
# Prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are a helpful assistant with two tools: (1) web search, "
         "(2) a custom, magic function. Use web search for current events and use "
         "magic_function if the question directly asks for it. Otherwise, answer directly. "
         "Current time: {time}."),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

# LLM chain
llm = ChatGroq(temperature=0, model="llama3-70b-8192")
assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools)

# %%
question = "What is magic_function(3)"
payload = assistant_runnable.invoke({"messages": [("user", question)]})
payload.tool_calls

# %%
question = "What is the capital of the US?"
payload = assistant_runnable.invoke({"messages": [("user", question)]})
payload.tool_calls

# %%
payload

# %% [markdown]
# ## Memory

# %%
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# %%
# SQL toolkit
#toolkit = SQLDatabaseToolkit(db='test.db', llm=llm)
#tools = toolkit.get_tools()

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            # Invoke the tool-calling LLM
            result = self.runnable.invoke(state)
            # If it is a tool call -> response is valid
            # If it has meaningful text -> response is valid
            # Otherwise, we re-prompt it because response is not meaningful
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                # Then we re-try
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# Tool
def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

# Utilities
def _print_event(event: dict, _printed: set, max_length=1500) -> None:
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"]
            ) for tc in tool_calls
        ]
    }


# Graph
builder = StateGraph(State)

builder.add_node("assistant", RunnableLambda(Assistant(assistant_runnable)))
builder.add_node("tools", create_tool_node_with_fallback(tools))

# Define edges: these determine how the control flow moves
builder.set_entry_point("assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is not a tool call -> tools_condition routes to END
    tools_condition,
    # 'Tools' calls one of our tools. END causses the graph to terminate(and repond to the user)
        {"tools": "tools", END: END},
)
builder.add_edge("tools", "assistant")


# The checkpointer lets the graph persist its state
memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory)


try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception as e:
    print(f"Failed to display graph: {e}")

# %%
questions = [
    "What is magic_function(3)",
    "What is the weather in SF now?",
]

_printed = set()
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id,  # Checkpoints are accessed by thread_id
    }
}

events = graph.stream(
    {"messages": [("user", questions[1])]}, config, stream_mode="values"
)

for event in events:
    _print_event(event, _printed)



