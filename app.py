import os
import streamlit as st
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
import textwrap
from retrieval.reranker import *
from retrieval.retriever import *
# Ensure you have a file named ingestion/neo4j_loader.py with connect_to_neo4j
from ingestion.neo4j_loader import connect_to_neo4j

# --- 1. Caching Resource-Intensive Objects ---

@st.cache_resource
def get_neo4j_driver():
    """Initializes and caches the Neo4j database driver."""
    return connect_to_neo4j()

@st.cache_resource
def load_llm(api_key):
    """Initializes and caches the Gemini LLM client using the provided API key."""
    return ChatGoogleGenerativeAI(
            api_key=st.session_state.gemini_api_key,
            model="gemini-2.0-flash-lite",
            temperature=0.1,
            max_tokens=100000,
            timeout=None,
            max_retries=2,
        )

@st.cache_resource
def build_full_chain(_llm):
    """
    Builds and caches the entire LangChain pipeline.
    The underscore prefix on the arguments tells st.cache_resource to hash the object's identity,
    not its contents, which is correct for complex objects like LLM clients and drivers.
    """

    def get_schema_description():
        """
        Returns a string description of the Neo4j graph schema
        for grounding the LLM.
        """
        return """
        The Neo4j database contains information about academic papers.
        The node labels are: Paper, Author, Category, Model, Dataset, Metrics, Libraries, Tasks, Concepts, Institute, and Chunk.
        Relationships describe how these entities are connected, for example:
        - (Paper)-[:WRITTEN_BY]->(Author)
        - (Paper)-[:IN_CATEGORY]->(Category)
        - (Paper)-[:MODEL_ALGORITHM_USED]->(Model)
        - (Paper)-[:DATASET_USED]->(Dataset)
        - (Paper)-[:METRICS_USED]->(Metrics)
        - (Paper)-[:LIBRARY_FRAMEWORK_USED]->(Libraries)
        - (Paper)-[:TASK_PERFORMED]->(Tasks)
        - (Paper)-[:THEORIES_CONCEPTS_USED]->(Concepts)
        - (Paper)-[:INSTITUTE]->(Institute)
        The 'Paper' node has a 'published' property which is a datetime object.

        You can generate plots for:
        1. The distribution of papers across different 'Category' nodes.
        2. The number of papers published per year (by extracting the year from the 'published' property).
        3. The count of papers using specific 'Model', 'Dataset', or 'Libraries'.
        Do not attempt to plot data that does not exist, like 'model accuracy scores' or 'citation counts'.
        """

    # --- 1. Define Output Parsers for Structured Output ---
    class PlotResponse(BaseModel):
        """Structured response for plot generation."""
        text_answer: str = Field(description="A textual answer to the user's query, summarizing the findings.")
        python_code: str = Field(description="A string containing a Python function `generate_plot(driver)` that uses plotly to create a histogram and returns a figure object.")

    class TextResponse(BaseModel):
        """Structured response for a text-only answer."""
        text_answer: str = Field(description="A textual answer to the user's query based on the retrieved context.")
        python_code: str = Field(description="An empty string or None, as no plot is generated.", default=None)

    # --- 2. Define the Router and its Logic ---
    routing_prompt_template = """
    Given the user query and the database schema, determine if a data visualization like a histogram is an appropriate response.
    The user might ask for a 'plot', 'distribution', 'histogram', 'chart', or ask about trends over time or frequencies of categories.

    Schema Description:
    {schema}

    User Query:
    {input}

    Respond with "plot" if a visualization is appropriate and possible with the given schema. Otherwise, respond with "text".
    """
    routing_prompt = ChatPromptTemplate.from_template(routing_prompt_template)
    schema_description = get_schema_description()
    router_chain = (
        {"input": lambda x: x['input'], "schema": lambda x: schema_description, "chat_history": lambda x: x.get("chat_history", [])}
        | routing_prompt
        | llm
        | (lambda x: "plot" if "plot" in x.content.lower() else "text")
    )

    plot_generation_prompt_template = """
    You are an expert Neo4j and Python developer tasked with generating a textual answer and a Python script for data visualization.
    Based on the user's query and the database schema, provide a comprehensive text answer and a Python function to generate a histogram using Plotly.
    Schema Description: {schema}
    User Query: {input}
    **Instructions:**
    1.  Provide a clear, text-based answer to the user's query.
    2.  Write a complete Python function `generate_plot(driver)` that:
        a. Includes its own necessary imports (like pandas and plotly.express) inside the function body.
        b. Contains a Cypher query to fetch the necessary data from the Neo4j database.
        c. **CRUCIAL:** You must process the `session.run(query)` result and convert it to a list **inside** the `with driver.session() as session:` block. This is required to prevent a "Result consumed" error.
        d. Processes the collected list of data (e.g., using pandas).
        e. Creates a histogram using `plotly.express` and returns the figure object.

    Return ONLY a JSON object with keys "text_answer" and "python_code".
    """
    plot_generation_prompt = ChatPromptTemplate.from_template(plot_generation_prompt_template)
    plot_chain = (
        {"input": lambda x: x['input'], "schema": lambda x: schema_description}
        | plot_generation_prompt
        | llm
        | JsonOutputParser(pydantic_object=PlotResponse)
    )

    # --- 4. Define the Standard RAG Chain (Modified for consistent output) ---
    custom_retriever_runnable = RunnableLambda(get_custom_retriever_documents)

    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Do not include any of the conversation history in your response, just the query itself. The query should be in a standalone format."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert academic research assistant. Answer the user's question based ONLY on the following pieces of retrieved context:\n\n{context}\n\nDo not make up information. Always cite the paper's title or entry_id at the end of each sentence or paragraph where you used information from that paper."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, custom_retriever_runnable, rephrase_prompt)
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    def format_rag_output(rag_output):
        return {"text_answer": rag_output["answer"], "python_code": None}

    text_chain = rag_chain | RunnableLambda(format_rag_output)


    # --- 5. Combine Chains with RunnableBranch ---
    branched_chain = RunnableBranch(
        (lambda x: x["topic"] == "plot", plot_chain),
        text_chain,
    )

    # CORRECTED LOGIC: Use RunnablePassthrough.assign to add the 'topic' key
    # without removing 'chat_history'. This is the fix for the KeyError.
    full_chain = RunnablePassthrough.assign(
        topic=router_chain,
        chat_history=lambda x: x.get("chat_history", [])
    ) | branched_chain


    # --- 6. Add Conversational History ---
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    return RunnableWithMessageHistory(
        full_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


# --- 2. Streamlit App Interface ---

st.set_page_config(page_title="Arxiv Trend Analyser", layout="wide")
st.title("Arxiv Trend Analyser 📈")

# --- API Key and Session Management in Sidebar ---
with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("Enter your Gemini API Key:", type="password")
    if st.button("Submit API Key"):
        if api_key_input and api_key_input.strip():
            st.session_state.gemini_api_key = api_key_input.strip()
            st.success("✅ API Key saved! You can now enter your queries.")
        else:
            st.error("❌ Please provide a valid API Key.")

    session_id = st.text_input("Session ID", value="default_session")

# Initialize session state for chat history and message store
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'store' not in st.session_state:
    st.session_state.store = {}

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "figure" in message:
            st.plotly_chart(message["figure"], use_container_width=True)


# --- Main Logic ---
if not st.session_state.get("gemini_api_key"):
    st.info("Please enter your Gemini API Key in the sidebar to begin.")
    st.stop()

try:
    # Load all cached resources
    driver = get_neo4j_driver()
    llm = load_llm(st.session_state.gemini_api_key)
    conversational_chain = build_full_chain(llm)

    # Use st.chat_input for a better user experience
    if prompt := st.chat_input("Ask about Arxiv trends, research papers, or authors..."):
        # Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process and display assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = conversational_chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": session_id}}
                )

                # Display the text answer
                st.markdown(response['text_answer'])
                assistant_message = {"role": "assistant", "content": response['text_answer']}

                # If a plot is generated, execute the code and display the plot
                if response.get('python_code'):
                    st.write("### Generated Visualization")
                    try:
                        full_code = f"""
import pandas as pd
import plotly.express as px
from neo4j import GraphDatabase

{response['python_code']}
                        """
                        # exec(full_code, globals(), local_scope)
                        full_code = textwrap.dedent(full_code)
                        exec(full_code, globals())
                        # plot_func = local_scope.get('generate_plot')
                        plot_func = globals().get('generate_plot')
                        
                        if callable(plot_func):
                            fig = plot_func(driver)
                            st.plotly_chart(fig, use_container_width=True)
                            assistant_message["figure"] = fig # Save figure to display on rerun
                        else:
                            st.error("Could not find a callable `generate_plot` function in the generated code.")
                            st.code(response['python_code'], language='python')
                    
                    except Exception as e:
                        st.error(f"An error occurred while generating the plot: {e}")
                        st.code(response['python_code'], language='python')
                
                st.session_state.messages.append(assistant_message)

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")