from gptClient import GPTClient
from config import Config
import prompt

class Validator:
    def __init__(self):
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )     

    async def validate(self, task_obj, result):
        print('------VALIDATE ONE TIME------')

        system_content = prompt.VALIDATION_PROMPT
        user_content = f'''
            Here is the subtask: {task_obj}
            Here is the result: {result}
        '''

        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': user_content}
        ]

        feedback = await self.gpt_client.a_chat_completion(messages, temperature=Config.TEMPERATURE)
        
        if result == "NONE":
            return None
        else:
            return feedback
    
    async def is_python_code(self, result):
        print('------CHECK IF PYTHON ONE TIME------')
        system_content = prompt.IS_PYTHON_PROMPT
        user_content = f'''
            Here is the result: {result}
        '''
        
        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': user_content}
        ]

        feedback = await self.gpt_client.a_chat_completion(messages, temperature=Config.TEMPERATURE)
        
        if result == "No":
            return None
        else:
            return feedback
    
    async def execute_python_code(self, task_obj, result):
        print('------CHECK PYTHON RESULT ONE TIME------')
        system_content = prompt.PYTHON_RESULT_PROMPT
        user_content = f'''
            Here is the subtask: {task_obj}
            Here is the result: {result}
        '''

        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': user_content}
        ]

        feedback = await self.gpt_client.a_chat_completion(messages, temperature=Config.TEMPERATURE)
        
        if result == "NONE":
            return None
        else:
            return feedback