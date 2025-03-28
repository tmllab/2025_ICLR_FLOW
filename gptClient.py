import asyncio
from openai import OpenAI, AsyncOpenAI
from typing import Dict, Any, List
from config import Config

# -----------------------------------------------------------------------------
# GPT Client
# -----------------------------------------------------------------------------
class GPTClient:
    """
    Wrapper around the OpenAI API for making GPT model calls.

    This class provides both synchronous and asynchronous interfaces for
    interacting with OpenAI's GPT models. It handles API configuration
    and message formatting.

    Attributes:
        api_key (str): OpenAI API key for authentication.
        model (str): GPT model identifier to use for completions.
        temperature (float): Default sampling temperature for responses.
        client (OpenAI): Synchronous OpenAI client instance.
        a_client (AsyncOpenAI): Asynchronous OpenAI client instance.
    """

    def __init__(self, api_key: str = Config.OPENAI_API_KEY, model: str = Config.GPT_MODEL, temperature: float = Config.TEMPERATURE):
        """
        Initialize GPTClient with API configuration.

        Args:
            api_key (str, optional): OpenAI API key. Defaults to Config.OPENAI_API_KEY.
            model (str, optional): GPT model to use. Defaults to Config.GPT_MODEL.
            temperature (float, optional): Sampling temperature. Defaults to Config.TEMPERATURE.
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key)
        self.a_client = AsyncOpenAI(api_key=self.api_key)
        
    async def a_chat_completion(self, messages: List[Dict[str, Any]], temperature: float = None) -> str:
        """
        Asynchronously generate a chat completion response.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries with 'role' and 'content'.
            temperature (float, optional): Override default temperature. Defaults to None.

        Returns:
            str: Generated response content from the model.

        Note:
            Each message dictionary should have the format:
            {'role': 'user'|'system'|'assistant', 'content': 'message text'}
        """
        temp = temperature if temperature is not None else self.temperature
        response = await self.a_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp
        )
        return response.choices[0].message.content
    
    def chat_completion(self, messages: List[Dict[str, Any]], temperature: float = None) -> str:
        """
        Synchronously generate a chat completion response.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries with 'role' and 'content'.
            temperature (float, optional): Override default temperature. Defaults to None.

        Returns:
            str: Generated response content from the model.

        Note:
            Each message dictionary should have the format:
            {'role': 'user'|'system'|'assistant', 'content': 'message text'}
        """
        temp = temperature if temperature is not None else self.temperature
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp
        )
        return response.choices[0].message.content

async def print_response(task):
    """
    Utility function to print responses from async tasks as they complete.

    Args:
        task (asyncio.Task): Asynchronous task that will return a response string.
    """
    response = await task
    print(response)  # Print as soon as each task finishes

async def main():
    """
    Example main function demonstrating parallel GPT API calls.

    This function shows how to:
    1. Create a GPTClient instance
    2. Make multiple async requests
    3. Handle responses asynchronously
    4. Run tasks in parallel

    Note:
        This is primarily for demonstration and testing purposes.
    """
    gptClient = GPTClient()
    messages = [ {'role': 'user', 'content': "say yes only"}  ]
    messages2 = [ {'role': 'user', 'content': "help me write equations for RL"}  ]

    task1 = asyncio.create_task(gptClient.a_chat_completion(messages2))
    task2 = asyncio.create_task(gptClient.a_chat_completion(messages))

    # Run both tasks without waiting for order
    asyncio.create_task(print_response(task1))
    asyncio.create_task(print_response(task2))
    
    await asyncio.sleep(100)

if __name__ == "__main__":
    asyncio.run(main())

