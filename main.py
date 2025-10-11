from langchain_community.document_loaders import WebBaseLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os



load_dotenv()

# LangSmith Ke Liye
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'Know-Darbhanga'
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['QDRANT_API_KEY'] = os.getenv('QDRANT_API_KEY')

# Loders
textLoader = TextLoader('./data/notes.txt')
webLoader = WebBaseLoader('https://en.wikipedia.org/wiki/Darbhanga')
textDocs = textLoader.load()
webDocs = webLoader.load()
full_docs = textDocs + webDocs
spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_docs = spliter.split_documents(full_docs)



# vector-embeddings
url = "https://b19d052c-1e20-484c-9492-758cfde64d6e.us-east-1-1.aws.cloud.qdrant.io:6333"
api_key = os.getenv('QDRANT_API_KEY')
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorStore = QdrantVectorStore.from_documents(
    documents=splitted_docs,
    embedding=embeddings,
    url=url,
    api_key=api_key,
    collection_name="know-darbhanga",
    force_recreate=True,
    timeout=60
)
retiever = vectorStore.as_retriever(search_kwargs={"k":5})

# model-llm
prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant.your task is to answer from the context given.if not found in the context then answer "I dont able to find the answer from the context".
    ###CONTEXT
    {context}
    Question: {input}
    Answer:
    """
)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
)
op_parser = StrOutputParser()
stuff_chain = create_stuff_documents_chain(llm,prompt)
retrival_chain = create_retrieval_chain(retriever=retiever,combine_docs_chain=stuff_chain)
while True:
    question = input("Enter your question: ")
    if question == "exit":
        break
    result = retrival_chain.invoke({"input": question})
    print('🔥Result -> ',result['answer'])


