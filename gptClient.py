from openai import OpenAI, AsyncOpenAI
from typing import Dict, Any, List
from config import Config

# -----------------------------------------------------------------------------
# GPT Client
# -----------------------------------------------------------------------------
class GPTClient:
    """Wrapper around the OpenAI API for GPT calls."""
    def __init__(self, api_key: str, model: str = Config.GPT_MODEL, temperature: float = Config.TEMPERATURE):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key)
        self.a_client = AsyncOpenAI(api_key=self.api_key)
        
    async def a_chat_completion(self, messages: List[Dict[str, Any]], temperature: float = None) -> str:
        temp = temperature if temperature is not None else self.temperature
        response = await self.a_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp
        )
        return response.choices[0].message.content
    
  
    def chat_completion(self, messages: List[Dict[str, Any]], temperature: float = None) -> str:

        temp = temperature if temperature is not None else self.temperature
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp
        )
        return response.choices[0].message.content
    

    


