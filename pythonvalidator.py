from gptClient import GPTClient
from config import Config
import prompt
import sys
import io
import asyncio

class PythonValidator:
    """Executes an individual task asynchronously via GPT."""
    def __init__(self, overall_task: str, max_itt):
        self.overall_task = overall_task
        ## TODO delete gpt_client
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )
        ## self.system_prompt = 
        
    def run(self, data):
        messages = {}
        
        self.gpt_client.chat_completion(messages=messages)

    def is_python():
        pass