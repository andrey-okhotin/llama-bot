import os

from langchain.document_loaders import DirectoryLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnablePassthrough

from fastapi_async_langchain.responses import StreamingResponse
from fastapi import FastAPI

from dotenv import load_dotenv
from pydantic import BaseModel




loaders = {
	'.pdf': PyPDFLoader,
	'.csv': CSVLoader
}
def create_directory_loader(file_type):
	return DirectoryLoader(
		path=os.path.join('app', 'data'),
		glob=f"**/*{file_type}",
		loader_cls=loaders[file_type],
	)
pdf_loader = create_directory_loader('.pdf')
csv_loader = create_directory_loader('.csv')
pdf_documents = pdf_loader.load()
csv_documents = csv_loader.load()
documents = pdf_documents + csv_documents

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(documents)

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
	model_name=model_name,
	model_kwargs=model_kwargs,
	encode_kwargs=encode_kwargs,
	cache_folder=os.path.join('app', 'embed_search_model')
)
vectorstore = Chroma.from_documents(documents=splits, embedding=hf)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

template = """
	Ты отвечаешь на вопросы пользователей, используя информацию из документов.
	Информация из документов: {information}.
	Вопрос {question}? Ответ на этот вопрос будет таким: 
"""
prompt = PromptTemplate(template=template, input_variables=["question", "information"])

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
	model_path=os.path.join('app', 'llama-2-7b-chat.Q4_K_M.gguf'),
	temperature=0.75,
	top_p=1,
	max_tokens=256,
	n_ctx=1024,
	callback_manager=callback_manager,
	verbose=False
)
qa_template = { 
	"information" : retriever, 
	"question": RunnablePassthrough()
}
rag_chain = (qa_template | prompt | llm)

app = FastAPI()

class Message(BaseModel):
	message: str
	user_id: str

@app.post("/message")
async def llama_read_and_write(message: Message):
	return rag_chain.invoke(message.message)
