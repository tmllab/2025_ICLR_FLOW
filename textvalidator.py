from gptClient import GPTClient
from config import Config
import json
import prompt

class textValidator:
    """
    A validator class that checks text-based task results using GPT.

    This class is responsible for:
    - Validating text-based task outputs
    - Providing feedback on task results
    - Determining if task requirements are met
    - Logging validation results

    Attributes:
        gpt_client (GPTClient): Client for interacting with GPT API.
        text_validation_prompt (str): Prompt template for text validation.
    """

    def __init__(self):
        """
        Initialize the textValidator with GPT client and validation prompt.

        Sets up:
        - GPT client with configuration parameters
        - Text validation prompt template
        """
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.text_validation_prompt = prompt.TEXT_VALIDATION_PROMPT

    async def validate(self, task_obj, result, history) -> tuple[str, str]:
        """
        Validate text-based task results using GPT.

        This method:
        1. Constructs a validation prompt with task requirements, history, and result
        2. Uses GPT to evaluate the result
        3. Logs validation results
        4. Determines validation status

        Args:
            task_obj: The task object containing validation requirements.
            result (str): The text result to validate.
            history (str): Previous execution history and feedback.

        Returns:
            tuple[str, str]: A tuple containing (feedback, status).
                           status can be 'completed' or 'failed'.
                           feedback is None if validation succeeds,
                           otherwise contains improvement suggestions.

        Note:
            The validation is considered successful if GPT responds with 'OK'
            in a short message (less than 50 characters).
        """

        user_content = f"""
## Current Task Requirement:
{task_obj}

---

## Current Task change History:
{history}

---

## Current Task Latest Result:
{result}
        """

        messages = [
            {'role': 'system', 'content': self.text_validation_prompt},
            {'role': 'user', 'content': user_content}
        ]
    
        feedback = await self.gpt_client.a_chat_completion(messages, temperature=Config.TEMPERATURE)
        with open('validate_log.json', 'a', encoding='utf-8') as file:
            file.write('----------\nGOT TEXT VALIDATION\n----------')
            json.dump({'task_obj': task_obj, 'result': result, 'feedback': feedback}, file, indent=4)

        
        if "OK" in feedback and len(feedback)<50:
            return "", 'completed'
        else:
            return feedback, 'failed'
    
