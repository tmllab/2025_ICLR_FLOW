import asyncio
import os
from openai import OpenAI
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

    async def a_chat_completion(self, messages: List[Dict[str, Any]], temperature: float = None) -> str:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.chat_completion(messages, temperature)
        )
        return response
    
  
    def chat_completion(self, messages: List[Dict[str, Any]], temperature: float = None) -> str:

        temp = temperature if temperature is not None else self.temperature
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp
        )
        return response.choices[0].message.content