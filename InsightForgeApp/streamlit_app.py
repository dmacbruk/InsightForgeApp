import os
import streamlit as st
import pandas as pd
import openai

# Attempt to load OpenAI API key from Streamlit secrets (Streamlit Cloud) or local secrets.toml
openai_api_key = None
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    # Fallback: try to read from local .streamlit/secrets.toml
    try:
        import toml
        secrets = toml.load(".streamlit/secrets.toml")
        openai_api_key = secrets.get("OPENAI_API_KEY")
    except Exception:
        openai_api_key = None

# If API key is still not found, warn the user and stop the app
if not openai_api_key:
    st.warning("**OpenAI API key not found!** 📌 Please add your API key to `st.secrets` in Streamlit Cloud (via *App Settings* -> *Secrets*) or to a local `.streamlit/secrets.toml` file if running the app locally.")
    st.stop()
# Set the API key for the OpenAI library
openai.api_key = openai_api_key

# Now import LangChain components (after setting the API key)
try:
    # For newer LangChain versions, ChatOpenAI is recommended
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
except ImportError:
    # Fallback to older LangChain LLM class if needed
    from langchain.llms import OpenAI
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# App title and description
st.title("InsightForge: AI-Powered Data Explorer")
st.markdown("Upload a CSV and ask questions to gain insights from your data. Powered by LangChain and OpenAI's GPT.")

# File uploader for CSV data
uploaded_file = st.file_uploader("**Upload a CSV file**", type=["csv"])
if uploaded_file is not None:
    # If a file is uploaded, read it into a DataFrame
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded **{uploaded_file.name}** successfully!")
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        st.stop()
else:
    # If no file uploaded, try using a default sample CSV (if available in the repo)
    sample_path = "sales_data.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        st.info("Using sample dataset **sales_data.csv** (since no file was uploaded).")
    else:
        st.warning("Please upload a CSV file to get started.")
        st.stop()

# Create a LangChain agent that can interact with the DataFrame
try:
    agent = create_pandas_dataframe_agent(llm, df, verbose=False)
except Exception as e:
    st.error(f"Failed to initialize the data agent: {e}")
    st.stop()

# User input for questions
query = st.text_input("💡 Ask a question about your data and press Enter:")
if query:
    with st.spinner("Analyzing data, please wait..."):
        try:
            # Run the agent to get an answer
            answer = agent.run(query)
            st.subheader("Answer")
            st.write(answer)
        except Exception as err:
            st.error(f"🚨 Sorry, I ran into an error: {err}")
