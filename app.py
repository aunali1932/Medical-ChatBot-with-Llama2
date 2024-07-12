from flask import Flask,request,render_template,jsonify
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from src.prompt import *
from src.helper import download_Hugging_face_Embeddings
# from store_index import docsearch

app= Flask(__name__)

embeddings=download_Hugging_face_Embeddings()
persist_directory = 'knowledge-base'
docsearch = Chroma(persist_directory=persist_directory,embedding_function= embeddings)

PROMPT = PromptTemplate(template=prompt_template,input_variables=["context","question"])

chain_arguments = {"prompt":PROMPT}
llm = CTransformers(model='model\llama-2-7b-chat.ggmlv3.q4_0.bin',model_type='llama',config={'temperature':0.7 })

qa = RetrievalQA.from_chain_type(llm=llm,retriever=docsearch.as_retriever(search_kwargs={"k": 2}),chain_type_kwargs=chain_arguments,return_source_documents=True)

@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/get',methods=['GET','POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    response = qa({'query': input})
    print(response)
    return str(response["result"])


if __name__ == '__main__':
    app.run(debug=True) 