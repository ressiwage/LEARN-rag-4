from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, get_response_synthesizer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.node_parser import SentenceSplitter
import ollama
from llama_index.core import StorageContext
from IPython.display import Markdown, display
import chromadb

Settings.llm=None

parser = SentenceSplitter()

embed_model =  HuggingFaceEmbedding(model_name="cointegrated/rubert-tiny2")

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

documents = SimpleDirectoryReader("./data/").load_data()
nodes = parser.get_nodes_from_documents(documents)

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model, transformations=[SentenceSplitter(chunk_size=200, chunk_overlap=0)]
)
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)
response_synthesizer = get_response_synthesizer()

QUERY = 'что меня ждет в этой квартире?'

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor()],
)
response = query_engine.query(QUERY)
# print(response.source_nodes)


# Query Data
content=f'ответь на русском языке с использованием контекста на следующий вопрос: {QUERY}; контекст: {"; ".join([i.text for i in response.source_nodes])}'
# print(content)
response = ollama.chat(model='llama2', stream=True, messages=[
  {
    'role': 'user',
    'content': content,
  },
  
])
for chunk in response:
  print(chunk['message']['content'], end='', flush=True)