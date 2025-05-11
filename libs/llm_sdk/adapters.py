import os
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from .base import LLMAdapter
from .types import ResponseType

load_dotenv(override=True)

class OpenAIAdapter(LLMAdapter):
    def __init__(self, base_url: str, model: str ):
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL",base_url)
        )

    def upload_file(self, file_path: str) -> str:
        return self.client.files.create(file=Path(file_path), purpose="batch").id

    def create_batch_job(self, input_file_id: str) -> str:
        """Creates a batch job using the LLM (currently only tested with Aliyun's platform). 
        Recommended for use when operating with limited budget.

        Args:
            input_file_id (str): The ID of the file to be processed in the batch job.
            
        Returns:
            str: The ID of the created batch job.
        """
        print(f"Creating batch job with input file ID: {input_file_id}")
        return self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        ).id

    def check_job_status(self, batch_id: str) -> str:
        return self.client.batches.retrieve(batch_id=batch_id).status

    def get_output_id(self, batch_id: str) -> str:
        return self.client.batches.retrieve(batch_id=batch_id).output_file_id

    def get_error_id(self, batch_id: str) -> str:
        return self.client.batches.retrieve(batch_id=batch_id).error_file_id

    def download_results(self, output_file_id: str, output_file_path: str):
        content = self.client.files.content(output_file_id)
        content.write_to_file(output_file_path)

    def download_errors(self, error_file_id: str, error_file_path: str):
        content = self.client.files.content(error_file_id)
        content.write_to_file(error_file_path)

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Chat with the LLM (maintains the conversation history)
        Args:
            messages (list[dict[str, str]]): A list of message dictionaries containing role and content.
            **kwargs: Additional parameters for the chat completion request (other params you can pass to the api).
        Returns:
            str: The response from the LLM.
        """        
        response = self.client.chat.completions.create(
            model=os.getenv("AGENT_MODEL_PATH",self.model),
            messages=messages,
            **kwargs 
        )
        return response.choices[0].message.content
    
    def generate(self, messages: list[dict[str, str]| str],type:ResponseType='text',**kwargs) -> str:
        """Generate text from the LLM (do not maintain the conversation history)
        Args:
            messages (list[dict[str, str]]): A list of message dictionaries containing role and content.
            type (ResponseType): The type of response to generate (default is 'text').
            **kwargs: Additional parameters for the chat completion request (other params you can pass to the api).
        Returns:
            str: The response from the LLM. 
        """
        response = self.client.chat.completions.create(
        model=os.getenv("AGENT_MODEL_PATH",self.model),
        messages=messages,
        **kwargs 
        )
        if len(response.choices) > 1:
            return [choice.message.content for choice in response.choices]
        return response.choices[0].message.content
        
class GoogleAdapter(LLMAdapter):
    """Not implemented yet, you can try yourself"""
    def __init__(self):
        raise NotImplementedError("Google Gemini API not yet implemented.")

    def upload_file(self, file_path): raise NotImplementedError()
    def create_batch_job(self, input_file_id): raise NotImplementedError()
    def check_job_status(self, batch_id): raise NotImplementedError()
    def get_output_id(self, batch_id): raise NotImplementedError()
    def get_error_id(self, batch_id): raise NotImplementedError()
    def download_results(self, output_file_id, output_file_path): raise NotImplementedError()
    def download_errors(self, error_file_id, error_file_path): raise NotImplementedError()
    def chat(self, messages): raise NotImplementedError()
