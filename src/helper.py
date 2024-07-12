from langchain.document_loaders import PyMuPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings

def load_pdf(directory):
    loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=70)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_Hugging_face_Embeddings():
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings