
from gptClient import GPTClient
from config import Config
import prompt

class Summary:
    """Run a subtask using GPT."""

    def __init__(self):

        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.summary_prompt = prompt.RESULT_EXTRACT_PROMPT
        
    def summary(self, task, chathistory):
        user_content = f'''
            Here is the task description: {task}
            Here is the workflow for the task: {chathistory}
        '''

        messages = [
            {'role': 'system', 'content': self.summary_prompt},
            {'role': 'user', 'content': user_content}
        ]

        return self.gpt_client.chat_completion(messages)

