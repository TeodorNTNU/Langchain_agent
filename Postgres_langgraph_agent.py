import os
import sys
import uuid
from datetime import datetime
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from langchain_core.tools import tool
from langchain.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import CohereEmbeddings
from langchain_postgres import PGVector
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from postgres_saver import PostgresSaver, JsonAndBinarySerializer
from IPython.display import Image, display

# Load environment variables
load_dotenv()

# Define constants and keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CONNECTION_STRING = 'postgresql://postgres:test@localhost:5432/vector_db'
COLLECTION_NAME = 'state_of_union_vectors'
END = "end"

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Initialize Postgres database
db = PGVector(
    connection=CONNECTION_STRING,
    collection_name='state_of_union_vectors',
    embeddings=embeddings,
    use_jsonb=True
)

# Define tools
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

# Define Assistant class
class Assistant:
    def __init__(self, runnable):
        self.runnable = runnable

    def __call__(self, state, config):
        while True:
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content or
                (isinstance(result.content, list) and not result.content[0].get("text"))
            ):
                state["messages"].append(("user", "Respond with a real output."))
            else:
                state["messages"].append(("assistant", result.content))
                break
        return {"messages": state["messages"]}

# Function to handle tool errors
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            {"content": f"Error: {repr(error)}\n please fix your mistakes.", "tool_call_id": tc["id"]}
            for tc in tool_calls
        ]
    }

# Create a tool node with fallback
def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

# Define the prompt template
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

# Initialize the LLM chain
llm = ChatGroq(temperature=0, model="llama3-70b-8192")
assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools)

# Initialize connection pool and checkpoint
pool = ConnectionPool(
    conninfo=CONNECTION_STRING,
    max_size=20,
)
checkpoint = PostgresSaver(
    sync_connection=pool,
)
PostgresSaver.create_tables(connection=pool)

# Initialize state graph
builder = StateGraph(State)
builder.add_node("assistant", RunnableLambda(Assistant(assistant_runnable)))
builder.add_node("tools", create_tool_node_with_fallback(tools))

# Define graph edges
builder.set_entry_point("assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition(
        {"tools": "tools", END: END},
    )
)
builder.add_edge("tools", "assistant")

# Compile the graph with checkpointing
graph = builder.compile(checkpointer=checkpoint)

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception as e:
    print(f"Failed to display graph: {e}")

# Function to print events
def _print_event(event, _printed, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: {current_state[-1]}")
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

# Example usage
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        "thread_id": thread_id,
    }
}
questions = ["What is the capital of France?"]
_printed = set()

events = graph.stream(
    {"messages": [("user", questions[0])]}, config, stream_mode="values"
)

for event in events:
    _print_event(event, _printed)

# Additional example usage with multiple questions
questions = [
    "What is magic_function(3)",
    "What is the weather in SF now?",
]

for question in questions:
    events = graph.stream(
        {"messages": [("user", question)]}, config, stream_mode="values"
    )

    for event in events:
        _print_event(event, _printed)



