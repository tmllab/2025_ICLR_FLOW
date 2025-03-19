import os

class Config:
    """Centralized configuration parameters."""

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    

    if not OPENAI_API_KEY:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    GPT_MODEL: str = "gpt-4o-mini"
    TEMPERATURE: float = 1


