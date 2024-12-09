from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator

from components.set_index import get_index
from components.set_llm import set_llm
from dto.configs import QueryConfig

# create llm
config = QueryConfig()
set_llm(model=config.model_name)
# llm = OpenAI(model="gpt-4", temperature=0.0)
# set_llm()
# build index
# ...
vector_index = get_index(storage_path=config.storage_path,data_path=config.data_path, use_llamaparse=config.use_llama_parse)

# define evaluator
evaluator = FaithfulnessEvaluator(llm=Settings.llm)

# query index
query_engine = vector_index.as_query_engine()
response = query_engine.query(
    "What is stock price of Finserve"
)
print(response)
eval_result = evaluator.evaluate_response(response=response)
print(str(eval_result.passing))