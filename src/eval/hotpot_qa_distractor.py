import os.path

from llama_index.core.evaluation.benchmarks import HotpotQAEvaluator
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.core.embeddings import resolve_embed_model

from components.set_index import persist_storage, get_index
from components.set_llm import set_llm
from dir_configs import add_rootpath
from dto.configs import QueryConfig

# llm = OpenAI(model="gpt-3.5-turbo")
# embed_model = resolve_embed_model(
#     "local:sentence-transformers/all-MiniLM-L6-v2"
# )
config = QueryConfig(
    model_name='llama3.2',
    data_path='data/paul_gram_data',
    temperature=1
)
set_llm(model=config.model_name, temperature=config.temperature)

index = get_index(
    data_path=config.data_path,storage_path=config.storage_path,use_llamaparse=False
)

# storage_path = add_rootpath(config.storage_path)
# if os.path.exists(storage_path):
#     print('loading existing storage')
#     storage_context = StorageContext.from_defaults(persist_dir=storage_path)
#     index = load_index_from_storage(storage_context)
# else:
#     print('creating new storage')
#     index = VectorStoreIndex.from_documents(
#         [Document.example()], embed_model=Settings.embed_model, show_progress=True
#     )
#     persist_storage(storage_path=storage_path,index=index)


# engine = index.as_query_engine(llm=Settings.llm)

from llama_index.core.postprocessor import SentenceTransformerRerank

rerank = SentenceTransformerRerank(top_n=3)

engine = index.as_query_engine(
    llm=Settings.llm,
    node_postprocessors=[rerank],
)

HotpotQAEvaluator().run(engine, queries=10, show_result=True)


# HotpotQAEvaluator().run(engine, queries=5, show_result=True)