from gptClient import GPTClient
from config import Config
import prompt
import sys
import io

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
        
        if feedback == "NONE":
            return None
        else:
            return feedback
    
    async def is_python_code(self, result) -> bool:
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
        
        if feedback == "N":
            return False
        else:
            return True
    
    async def execute_python_code(self, code_str): 
        """Executes a Python script from a string and captures the output."""
        # Redirect stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        exec_globals = {"code": code_str}
        try:
            # Execute the provided code string
            exec(code_str, exec_globals)
        except Exception as e:
            print(f"Error executing code: {e}")
        finally:
            # Capture the output
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        
        return output