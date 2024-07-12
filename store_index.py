from src.helper import load_pdf
from src.helper import text_split,download_Hugging_face_Embeddings
from langchain.vectorstores import Chroma

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
print("Creating embeddings...")
embeddings=download_Hugging_face_Embeddings()
print("embeddings done")

persist_directory = 'knowledge-base'
docsearch = Chroma.from_documents(documents=text_chunks,embedding=embeddings,persist_directory=persist_directory)
