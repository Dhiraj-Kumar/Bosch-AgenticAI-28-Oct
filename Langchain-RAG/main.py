from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# -------------------------- Ingestion Process -------------------------------

# Step 1. Loading PDF Document
loader = PyPDFLoader("./Documents/HR-Policy.pdf")
documents = loader.load()

# Step 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

# Step 3. Create an instance of an embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# Step 4. Store embeddings into vector DB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chormadb"
)

# -------------------------- Retrieval Process ------------------------------
# Step 1. Create a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Step 2. Augmentation
prompt = PromptTemplate(
    template="""
    You are a helpful AI assistant. Answer the question using ONLY the context below
    Context:
    {context}
    Question:
    {question}
    """,
    input_variables=["context", "question"]
)

# Step 3. Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)

# Step 4. Create a Retrieval Pipeline
chain = (RunnableParallel(context=retriever, question=RunnablePassthrough())) | prompt  | llm | StrOutputParser() #LCEL

while True:
    user_input = input("You: ")
    if user_input.lower()=="exit":
        break
    answer = chain.invoke(user_input)
    print(f"AI: {answer}")