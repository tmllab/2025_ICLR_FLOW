from gptClient import GPTClient
from config import Config
import json
import prompt
import sys
import io

class pythonValidator:
    def __init__(self):
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.generate_test_prompt = prompt.TESTCODE_GENERATION_PROMPT
        self.check_python_prompt = prompt.IS_PYTHON_PROMPT
        
    async def validate(self, task_obj, result) -> str:
        '''Generate python test code and execute test code.'''
        print('------Run pythonValidator.validate()------')

        test_code = await self.generate_test_function(task_obj, result)
        runresult = await self.execute_python_code(test_code)

        # Logging execution result for debugging
        with open('validate_log.json', 'a', encoding='utf-8') as file:
            file.write('----------\nGOT PYTHON VALIDATION\n----------')
            json.dump({'task_obj': task_obj, 'result': result, 'testcode': test_code, 'runresult': runresult}, file, indent=4)

        print(f'***runresult is: {runresult}***')

        if 'Error executing code:' not in runresult:
            print('***Python code with no bugs***')
            return None, 'completed'
        else:
            print('***Python code with bugs***')
            return runresult, 'failed'
    
    async def generate_test_function(self, task_obj, result) -> str:
        '''Generate test function according to task objective and execute result.'''
        print('------Run pythonValidator.generate_test_function()------')

        user_content = f'''
            Here is the task object: {task_obj}
            Here is the result: {result}
        '''

        messages_exe = [
            {'role': 'system', 'content': self.generate_test_prompt},
            {'role': 'user', 'content': user_content}
        ]

        test_code = await self.gpt_client.a_chat_completion(messages_exe, temperature=Config.TEMPERATURE)
        test_code = test_code.strip('```python').strip('```')

        return test_code
    
    async def execute_python_code(self, test_code): 
        """Executes a Python script from a string and captures the output."""
        print('------Run pythonValidator.execute_python_code------')

        # Redirect stdout
        origin_stdout = sys.stdout
        sys.stdout = io.StringIO()
        exec_globals = {"code": test_code}

        try:
            # Execute the provided code string
            exec(test_code, exec_globals)
        except Exception as e:
            print(f"Error executing code: {e}")
        finally:
            # Capture the output
            output = sys.stdout.getvalue()
            sys.stdout = origin_stdout
        
        return output
    
    async def is_python_code(self, result,task_obj) -> bool:
        print('------Run pythonValidator.is_python_code()------')
        ##TODO check if the result need to be runned based on task_obj, 
        # it is possible that the code is just for explaination  
        user_content = f'''
            Here is the result: {result}
        '''
        
        messages = [
            {'role': 'system', 'content': self.check_python_prompt},
            {'role': 'user', 'content': user_content}
        ]
        print(user_content)
        
        feedback = await self.gpt_client.a_chat_completion(messages, temperature=Config.TEMPERATURE)
        print('------Is Python Code?:------', feedback)
        
        if feedback == "N":
            return False
        else:
            return True
    