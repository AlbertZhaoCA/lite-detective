import os
import asyncio
from tenacity import retry, stop_never, wait_exponential, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor
from .adapters import OpenAIAdapter, GoogleAdapter
from .types import Plateform, ResponseType

class LLM:
    def __init__(self, model:str, plateform: Plateform="OtherOpenAILikes",system_prompt: str = None, **kwargs):
        self.model = model
        self.adapter = self._init_adapter(plateform)
        self.system_prompt = system_prompt

    def _init_adapter(self, plateform: Plateform):
        match plateform:
            case "Aliyun":
                return OpenAIAdapter("https://dashscope.aliyuncs.com/compatible-mode/v1",self.model)
            case "Azure":
                return OpenAIAdapter(os.getenv("AZURE_OPENAI_ENDPOINT"))
            case "OpenAI":
                return OpenAIAdapter("https://api.openai.com/v1",self.model)
            case "Google":
                return GoogleAdapter()
            case "OtherOpenAILikes":
                return OpenAIAdapter("http://localhost:8000/v1",self.model)
            case _:
                raise ValueError(f"Unsupported platform: {plateform}")

    async def run_batch_job(self, input_file_path: str, output_file_path: str, error_file_path: str = None):
        input_id = self.adapter.upload_file(input_file_path)
        batch_id = self.adapter.create_batch_job(input_id)

        while True:
            status = self.adapter.check_job_status(batch_id)
            if status == "completed":
                print("Batch job completed successfully.")
                break
            elif status == "failed":
                batch = self.adapter.client.batches.retrieve(batch_id=batch_id)
                print(f"Batch job failed with error: {batch.errors}")
                print("Batch job failed.")
                raise RuntimeError("Batch job failed.")
            await asyncio.sleep(5)

        self.adapter.download_results(self.adapter.get_output_id(batch_id), output_file_path)
        if error_file_path:
            error_id = self.adapter.get_error_id(batch_id)
            if error_id:
                self.adapter.download_errors(error_id, error_file_path)

    @retry(
        stop=stop_never,
        wait=wait_exponential(multiplier=2, min=10, max=100),
        retry=retry_if_exception_type(Exception) 
    )
    def generate(self, messages: list[dict[str, str]| str],type:ResponseType='text',**kwargs) -> str:
        """generate response (do not maintain history)
        Args:
            messages(list[dict[str, str]| str]):message, follow system, user, assistant apprroach
            type('text' | 'json_object'): the type of the response
            **kwargs: other arguments you want to pass to the adapter
        """
        try:
            if isinstance(messages, str):
                if self.system_prompt:
                    messages = [
                        {
                            "role": "system",
                            "content": self.system_prompt
                        },
                        {
                            "role": "user",
                            "content": messages
                        }
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": messages
                        }
                    ]
            elif isinstance(messages, list):
                if self.system_prompt:
                    messages = [
                        {
                            "role": "system",
                            "content": self.system_prompt
                        }
                    ] + messages
                else:
                    messages = messages
            res = self.adapter.generate(messages, type=type, **kwargs)
        except Exception as e:
            print(f"Error generating response: {e}")
            raise         
        return res
    
    @retry(
        stop=stop_never,
        wait=wait_exponential(multiplier=2, min=10, max=100),
        retry=retry_if_exception_type(Exception)
    )
    def batch_generate(self, prompts: list[str], raw: bool = False, n: int = 1) -> list[str]:
        """generate response in batch (do not maintain history)
        @deprecated: use generate() instead
        Args:
            prompts(list[str]): The prompts.
            raw(bool): Whether to return the raw output.
            n(int): The number of responses to generate.
        Returns:
            The response from the LLM.
        """
        def generate_for_prompt(messages: str):
            if isinstance(messages, str):
                if self.system_prompt:
                    messages = [
                        {
                            "role": "system",
                            "content": self.system_prompt
                        },
                        {
                            "role": "user",
                            "content": messages
                        }
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": messages
                        }
                    ]
            elif isinstance(messages, list):
                if self.system_prompt:
                    messages = [
                        {
                            "role": "system",
                            "content": self.system_prompt
                        }
                    ] + messages
                else:
                    messages = messages
            try:
                res = self.adapter.generate(messages, type="text", n=n)
                return res
            except Exception as e:
                print(f"Error for prompt '{messages}': {e}")
                return None

        with ThreadPoolExecutor() as executor:
            return list(executor.map(generate_for_prompt, prompts))
        
    def chat(self, messages: list[dict[str, str]]) -> str:
        """not implemented yet"""
        try:
            res = self.adapter.chat(messages)
        except Exception as e:
            print(f"Error generating response: {e}")
            return
        return res
        


    
if __name__ == "__main__":
    llm = LLM("deepseek-cha",system_prompt="你是一个对话机器人")
    print(llm.generate("你是谁,以json格式输出",type="json_object"))
