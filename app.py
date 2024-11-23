# app.py

import os
import uuid
import streamlit as st
from typing import Optional

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.tools.retriever import create_retriever_tool

from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Set environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    st.error(
        "OPENAI_API_KEY not found. Please set it in your environment or .env file."
    )

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
    use_chat_groq = True
else:
    use_chat_groq = False
    # If you're not using ChatGroq, you can ignore this variable or set it accordingly

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
else:
    st.error(
        "LANGCHAIN_API_KEY not found. Please set it in your environment or .env file."
    )

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Demo Upwork"

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
gpt = "gpt-4o-mini"
llm = ChatOpenAI(model=gpt, temperature=0.2)
# Uncomment the following lines if you have access to ChatGroq and want to use it
# llama_3_2 = "llama-3.2-90b-vision-preview"
# llm = ChatGroq(model=llama_3_2, temperature=0.2)

persist_directory = "./data/chroma_langchain_db"
settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory,
)


# Function to process documents and get retriever
def process_docs_and_get_retriever(
    doc_path: str,
    persist_directory="./data/chroma_langchain_db",
    collection_name: str = "user_data",
    k: int = 4,
):
    if doc_path.endswith(".pdf"):
        try:
            loader = PyPDFLoader(doc_path)
            docs = loader.load_and_split()
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return None
    elif doc_path.endswith(".docx"):
        try:
            loader = Docx2txtLoader(doc_path)
            docs = loader.load()
        except Exception as e:
            print(f"Error loading DOCX: {e}")
            return None
    else:
        print("Unsupported file format. Please provide a .pdf or .docx file.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["# ", "## ", "### ", "\n\n", "\n- ", ". ", " "],
    )
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists(persist_directory):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name,
            client_settings=settings,
        )
        vector_store.add_documents(splits)
    else:
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
            client_settings=settings,
        )

    return "All Data added"


# Function to recreate retriever
def recreate_retriever(
    persist_directory: str = "./data/chroma_langchain_db",
    collection_name: str = "user_data",
):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return vector_store.as_retriever(search_kwargs={"k": 4})


# Initialize vector store
chroma_db_path = "./data/chroma_langchain_db"
if not os.path.exists(chroma_db_path):
    os.makedirs("./data/docs", exist_ok=True)
    doc_path = "./data/docs/goldmine.docx"
    retriever_doc = process_docs_and_get_retriever(doc_path)
    print("Retriever is ready:", retriever_doc)
else:
    print(f"Path already exists: {chroma_db_path}")

# Set up tools and retriever
tools = []
retriever = recreate_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "retriever_tool",
    "Search and return information for user query.",
)
tools.append(retriever_tool)


# Define State class
class State(MessagesState):
    query: str
    doc_path: Optional[str]


# Define vector_data function
def vector_data(state: State):
    if "doc_path" in state and state["doc_path"]:
        print("\n\n********** ADDING DATA TO VECTOR STORE **********\n\n")
        doc_path = state["doc_path"]
        process_docs_and_get_retriever(doc_path)
        return {
            "doc_path": "",
            "messages": [
                AIMessage(content="Document successfully added to the knowledge base.")
            ],
        }
    else:
        return {"doc_path": ""}


# Define call_model function
def call_model(state: State):
    messages = state.get("messages", [])
    query = state.get("query", "").strip()

    if not query:
        return {
            "messages": [
                AIMessage(
                    content="No query provided. You can now ask questions about the uploaded document."
                )
            ]
        }

    sys_msg = SystemMessage(
        content="""You are a helpful assistant.

Your main task is to assist the user by providing accurate information using the retriever_tool. Always answer based only on the information retrieved with the retriever_tool.

If you do not find the information needed to answer the user's question, clearly state: 
'I do not have the information you are looking for. If you just added new information, you might try asking your question again.' 
Then, politely ask the user to update the knowledge base with relevant details.

Do not provide answers based on assumptions, external knowledge, or unsupported information. Always rely solely on the retrieved content."""
    )

    message = HumanMessage(content=query)
    messages.append(message)
    print(f"WE ARE PASSING THIS TO THE LLM: \n\n*******\n{messages}\n*******\n\n")

    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke([sys_msg] + messages)

    return {"messages": response}


# Set up workflow
workflow = StateGraph(State)
workflow.add_node("vector_data", vector_data)
workflow.add_node("call_model", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("vector_data")
workflow.add_edge("vector_data", "call_model")
workflow.add_conditional_edges(
    "call_model", tools_condition, path_map=["tools", "__end__"]
)
workflow.add_edge("tools", "call_model")

memory = MemorySaver()
react_graph = workflow.compile(checkpointer=memory)


# Define call_graph function
def call_graph(user_input, config, doc_path=None):
    state = {}

    if user_input:
        state["query"] = user_input

    if doc_path:
        state["doc_path"] = doc_path

    events = react_graph.stream(state, config, stream_mode="values")

    response = None
    for event in events:
        if "messages" in event and event["messages"]:
            response = event["messages"][-1].content

    if doc_path and not user_input:
        response = "Document successfully added to the knowledge base."

    return response


# Streamlit app
def main():
    st.title("Multimodal Chatbot")
    st.write("Ask questions and optionally upload a document (PDF or DOCX).")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a document (PDF or DOCX):", type=["pdf", "docx"]
    )

    # Chat interface
    # Display chat messages from history on app rerun
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])

    # Accept user input
    user_input = st.chat_input("Your message")

    if user_input or uploaded_file is not None:
        # Prepare config
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

        # Save uploaded file to disk if any
        doc_path = None
        if uploaded_file is not None:
            os.makedirs("./data/docs", exist_ok=True)
            doc_path = os.path.join("./data/docs", uploaded_file.name)
            with open(doc_path, "wb") as f:
                f.write(uploaded_file.read())

        # Call the graph
        response = call_graph(user_input=user_input, config=config, doc_path=doc_path)

        # Update chat history
        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
        if response:
            st.session_state["messages"].append(
                {"role": "assistant", "content": response}
            )

        # Display the latest messages
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
        if response:
            with st.chat_message("assistant"):
                st.markdown(response)


if __name__ == "__main__":
    main()
