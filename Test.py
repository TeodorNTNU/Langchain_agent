import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.prebuilt import create_react_agent
from postgres_saver import PostgresSaver, JsonAndBinarySerializer
from IPython.display import Image, display
from langchain_postgres import PGVector
from langgraph.graph import END

#
# Load environment variables
load_dotenv()

# Define constants and keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CONNECTION_STRING = 'postgresql://postgres:test@localhost:5432/vector_db'
COLLECTION_NAME = 'state_of_union_vectors'

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

tools = [magic_function, web_search]

# Initialize the chat model
model = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)

# Create the REACT agent
graph = create_react_agent(model, tools=tools)

# Initialize connection pool and checkpoint
pool = ConnectionPool(
    conninfo=CONNECTION_STRING,
    max_size=20,
)
checkpoint = PostgresSaver(
    sync_connection=pool,
)
PostgresSaver.create_tables(pool)  # Ensure the tables are created

# Define initial state and config
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        "thread_id": thread_id,
    }
}

initial_state = {"messages": [("user", "How are you, what is the weather tomorrow?")]}

# Run the graph
try:
    for state in graph.stream(initial_state, config=config, stream_mode="values"):
        message = state["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            print(message.pretty_repr(html=True))
    print("Graph execution completed.")
except Exception as e:
    print(f"Graph execution failed: {e}")