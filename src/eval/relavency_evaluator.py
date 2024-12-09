from llama_index.core.evaluation import DatasetGenerator
from components.set_index import get_index, get_document_from_index
from components.set_llm import set_llm
from dto.configs import QueryConfig

config = QueryConfig(
    data_path='data/paul_gram_data'
)
set_llm()
index = get_index(
    data_path=config.data_path,
    storage_path=config.storage_path,
)

documents = get_document_from_index(
    index=index
)

assert isinstance(documents, list)
data_generator = DatasetGenerator.from_documents(documents)

eval_questions = data_generator.generate_questions_from_nodes()
print(eval_questions)

