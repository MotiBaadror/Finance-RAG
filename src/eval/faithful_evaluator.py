from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Response,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd

from components.set_llm import set_llm

# pd.set_option("display.max_colwidth", 0)# gpt-4
gpt4 = set_llm(model='llama3.2', temperature=0.75)

evaluator_gpt4 = FaithfulnessEvaluator(llm=gpt4)