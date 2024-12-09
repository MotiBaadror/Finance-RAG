from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def set_llm(model='llama3.2',temperature=0.75):
    print(f'Setting LLM model: {model}')
    Settings.llm = Ollama(model=model, temperature=temperature)

    Settings.embed_model = OllamaEmbedding(
        model_name=model,
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0},
    )
    print(Settings.llm.complete('Hi, How are you?'))

# if __name__ == '__main__':
#     set_llm()
#     Settings.llm.generate('Hi, I am good')
