from gptClient import GPTClient
import logging
from config import Config

class Validator:
    def __init__(self, max_itt):
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.max_itt = max_itt

    def call_validate(self, task_obj, result) -> list[str, str]:

        # loop for validation self.max_itt times
        i = 0
        while i < self.max_itt:
            validate_result, reason = self.validate(task_obj, result)
            if not reason:
                break
            i += 1

        # return reason and validation result.
        if i == self.max_itt:
            return reason, validate_result
        else:
            return None, validate_result

    def validate(task_obj, result):
        return