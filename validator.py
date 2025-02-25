from gptClient import GPTClient
from config import Config
import prompt
import sys
import io
import asyncio

class Validator:
    def __init__(self):
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )     

    async def validate(self, task_obj, result):
        # judge whether the result contains python code
        if self.is_python_code(result):
            # if is python code
            return self.python_validate(task_obj, result)

        # if not python code, execute general validation
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
            print('---no python code contained.---')
            return False
        else:
            print('---python code contained.---')
            return True
        
    async def python_validate(self, task_obj, result) -> str:
        # Generate python test code and execute test code.
        print('------RUN Python_validate function ONE TIME------')
        test_code = await self.generate_test_function(task_obj, result)
        runresult = await self.execute_python_code(test_code)

        print(f'***runresult is: {runresult}***')

        if 'Error executing code:' not in runresult:
            print('***Python code with no bugs***')
            return None
        print('***Python code with bugs***')

        return runresult
    
    async def generate_test_function(self, task_obj, result) -> str:
        # Generate test function according to task objective and execute result
        system_content_exe = prompt.TESTCODE_GENERATION_PROMPT
        # TODO: modify the prompt in line 79, task_obj is the task objective, LLM should know what this task for in order to generate a test code.
        user_content_exe = f'''
            Here is the task object: {task_obj}
            Here is the result: {result}
        '''
        messages_exe = [
            {'role': 'system', 'content': system_content_exe},
            {'role': 'user', 'content': user_content_exe}
        ]

        test_code = await self.gpt_client.a_chat_completion(messages_exe, temperature=Config.TEMPERATURE)
        test_code = test_code.strip('```python').strip('```')

        return test_code
    
    async def execute_python_code(self, test_code): 
        print('------RUN execute function to CHECK bugs ONE TIME------')
        
        print('------Generated Test Code is:------')
        print(test_code)

        """Executes a Python script from a string and captures the output."""
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
    
    




async def main():
    # Your main logic here
    print("Running main...")
    test_code = """
def add(a, b):
    return a * b  # Intentional bug

# We'll define a function that tests each assertion individually.
# Each assertion is wrapped in try/except to capture all failures.
def run_tests():
    failures = []

    try:
        assert add(2, 3) == 5, "Test failed: add(2,3) should return 5"
    except AssertionError as e:
        failures.append(str(e))

    try:
        assert add(-1, 1) == 0, "Test failed: add(-1,1) should return 0"
    except AssertionError as e:
        failures.append(str(e))

    try:
        assert add(0, 0) == 0, "Test failed: add(0,0) should return 0"
    except AssertionError as e:
        failures.append(str(e))

    if failures:
        for f in failures:
            print(f)
    else:
        print("All tests passed!")

# Execute our tests
run_tests()
        """
    val = Validator()
    result = await val.execute_python_code(test_code)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())