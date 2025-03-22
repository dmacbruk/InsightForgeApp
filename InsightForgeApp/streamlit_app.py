import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# If using Pandas DataFrame agent:
from langchain.agents import create_pandas_dataframe_agent

# Load OpenAI API key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set page config
st.set_page_config(page_title="InsightForge - BI Assistant", page_icon="📊", layout="wide")

st.title("InsightForge: AI-Powered Business Intelligence Dashboard")

# Load default dataset
default_df = pd.read_csv("sales_data.csv")
dataframes = [default_df]  # list to hold all datasets
# Upload Data section (sidebar nav will control visibility)
uploaded_file = None

# Sidebar navigation
section = st.sidebar.radio("Navigate", ["Upload Data", "Visualizations", "Chat Insights"])

if section == "Upload Data":
    st.header("📁 Upload Additional Dataset")
    st.write("Upload a CSV file to add its data to the analysis.")
    uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
    if uploaded_file:
        try:
            new_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully! Preview:")
            st.dataframe(new_df.head())
            # Append to existing data
            dataframes.append(new_df)
            st.write("New data has been added to the dataset for analysis.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Combine all data for analysis (default + any uploaded)
combined_df = pd.concat(dataframes, ignore_index=True)

if section == "Visualizations":
    st.header("📊 Data Visualizations and Insights")
    # Example 1: Sales over time line chart
    # Ensure Date is datetime
    if pd.api.types.is_string_dtype(combined_df['Date']):
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    sales_over_time = combined_df.groupby(pd.Grouper(key='Date', freq='M'))['Sales'].sum().reset_index()
    st.subheader("Sales Over Time")
    st.line_chart(sales_over_time, x="Date", y="Sales")
    # Example 2: Product performance bar chart
    prod_sales = combined_df.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
    st.subheader("Total Sales by Product")
    st.bar_chart(prod_sales, x="Product", y="Sales")
    # Example 3: Regional performance bar chart
    region_sales = combined_df.groupby('Region')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
    st.subheader("Total Sales by Region")
    st.bar_chart(region_sales, x="Region", y="Sales")
    # Example 4: Summary statistics
    st.subheader("Key Statistics")
    stats = combined_df[['Sales', 'Customer_Age', 'Customer_Satisfaction']].describe()
    st.table(stats)
    st.write("*Highlights:*")
    total_sales = combined_df['Sales'].sum()
    best_product = prod_sales.iloc[0]['Product']
    best_region = region_sales.iloc[0]['Region']
    avg_satisfaction = combined_df['Customer_Satisfaction'].mean()
    st.write(f"- **Total Sales:** {total_sales}")
    st.write(f"- **Best-Selling Product:** {best_product}")
    st.write(f"- **Top Region by Sales:** {best_region}")
    st.write(f"- **Average Customer Satisfaction:** {avg_satisfaction:.2f}")

if section == "Chat Insights":
    st.header("💬 Chat with Your Data")
    st.write("Ask any question about the business data, and InsightForge will answer with facts from the data.")
    # Initialize retriever and QA chain (do this once)
    # Create vector store from the combined data
    docs = []
    for _, row in combined_df.iterrows():
        # Simple document: join all fields into text
        doc_text = " | ".join(f"{col}: {row[col]}" for col in combined_df.columns)
        docs.append(doc_text)
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_texts(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    # Set up conversational QA chain with memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    # Alternatively, using Pandas DataFrame Agent:
    # llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model_name="gpt-3.5-turbo", temperature=0)
    # agent = create_pandas_dataframe_agent(llm, combined_df, verbose=False)
    # Chat interface
    if "history" not in st.session_state:
        st.session_state["history"] = []  # store conversation turns
    # Display prior chat messages
    for msg in st.session_state["history"]:
        role, content = msg["role"], msg["content"]
        st.chat_message(role).write(content)
    # Input box for new question
    user_query = st.chat_input("Your question")
    if user_query:
        # Display user message
        st.session_state["history"].append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
        # Get answer from chain
        result = qa_chain({"question": user_query})
        answer = result["answer"]
        # If using agent: answer = agent.run(user_query)
        st.session_state["history"].append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
