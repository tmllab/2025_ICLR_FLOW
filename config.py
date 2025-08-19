class Config:
    """Centralized static configuration parameters."""

    OPENAI_API_KEY: str = ""
    GPT_MODEL: str = "o4-mini"
    TEMPERATURE: float = 1

    @classmethod
    def set(cls, key: str, value):
        """Set a configuration parameter dynamically."""
        if hasattr(cls, key):
            setattr(cls, key, value)
        else:
            raise AttributeError(f"{key} is not a valid configuration attribute.")

    @classmethod
    def get(cls, key: str):
        """Retrieve a configuration parameter."""
        if hasattr(cls, key):
            return getattr(cls, key)
        else:
            raise AttributeError(f"{key} is not a valid configuration attribute.")
