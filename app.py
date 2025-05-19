import os
import bs4
import time
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import getpass
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain import hub
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph, MessagesState, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, END, StateGraph, MessagesState
from IPython.display import Image, display
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("RAG con LangGraph y Ollama")
st.caption("📄🤖 Chatbot RAG con archivo de texto")
st.caption("Carga un archivo y haz preguntas con RAG")

with st.sidebar:
    st.write('## Configuración de Parámetros')
    if st.button("🔄 Reiniciar conversación"):
        st.session_state.messages = []
        st.session_state.file_text = ""
        st.success("Chatbot reiniciado correctamente.")
        st.rerun()

    st.session_state.temperature = st.slider(
        'Temperatura',
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    st.session_state.top_p = st.slider(
        'Top P',
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.1
    )
    st.session_state.top_k = st.slider(
        'Top K',
        min_value=0,
        max_value=100,
        value=50,
        step=1
    )
    st.session_state.max_tokens = st.slider(
        'Max tokens',
        min_value=1,
        max_value=4096,
        value=256,
        step=1
    )

llm = ChatOllama(
    model = "qwen3:1.7b",
    temperature=st.session_state.temperature,
    top_p=st.session_state.top_p,
    top_k=st.session_state.top_k,
    num_predict=st.session_state.max_tokens
)

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

archivo = st.file_uploader("Sube un archivo de texto o como PDF)", type=["txt", "pdf"])

if archivo is not None:
    ruta_temporal = os.path.join("temp", archivo.name)
    os.makedirs("temp", exist_ok=True)
    with open(ruta_temporal, "wb") as f:
        f.write(archivo.read())
    
    if archivo.name.endswith(".pdf"):
        loader = PyPDFLoader(ruta_temporal)
    else:
        loader = TextLoader(ruta_temporal)

    docs = loader.load()
    st.write(f"La cantidad total de paginas del documento es: {len(docs)}")
    st.write(f"{docs[0].page_content[:300]}\n")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs_divididos = splitter.split_documents(docs)
    st.write(f"La cantidad de documentos despues de la separacion es : {len(docs_divididos)}")

    _ = vector_store.add_documents(documents=docs_divididos)
    st.success("✅ Documento cargado y procesado correctamente.")

graph_builder = StateGraph(MessagesState)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "You always answer questions in a humane, respectful and kind way."
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

if "messages" not in st.session_state:
    st.session_state.messages = []


query = st.chat_input("Haz una pregunta sobre el texto...")

if query:
    with st.chat_message("human"):
        st.markdown(query)
    st.session_state.messages.append(HumanMessage(content=query))


    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            result = graph.invoke({"messages": [{"role": "user", "content": query}]})
            respuesta = result["messages"][-1].content
        
        def stream_response(text):
            for line in text.splitlines():
                yield line + "\n"
                time.sleep(0.2)
        
        st.write_stream(stream_response(respuesta))
        st.session_state.messages.append(AIMessage(content=respuesta))
