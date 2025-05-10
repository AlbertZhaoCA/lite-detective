from .types import ModelType

class LLM:
     def __new__(cls, model: str = "deepseek-v3", type: ModelType = "local", **kwargs):
        if type == "local":
            from .local_llm import LLM as LocalLLM
            return LocalLLM(model,**kwargs)
        elif type == "ollama":
            from .ollama_llm import Ollama as OllamaLLM
            return OllamaLLM(model,**kwargs)
        elif type == "remote":
            from .remote_llm import LLM as RemoteLLM
            return RemoteLLM(model,**kwargs)
        else:
            raise ValueError(f"Unknown model type: {type}")
        
if __name__ == "__main__":
    llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct", type="local")
    print(llm.generate("I am albert"))
    llm = LLM(type="remote")
    print(llm.generate('how old are you, output should be json format, example: {"age": 18}',type="json_object"))
        
  
