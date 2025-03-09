from gptClient import GPTClient
from config import Config
import preprocessing
import json
import prompt
import sys
import io

class pythonValidator:
    """
    A validator class that checks Python code execution and validity.

    This class is responsible for:
    - Validating Python code execution
    - Generating test code for validation
    - Executing Python code safely
    - Checking if content contains executable Python code

    Attributes:
        gpt_client (GPTClient): Client for interacting with GPT API.
        generate_test_prompt (str): Prompt template for test code generation.
        check_python_prompt (str): Prompt template for Python code detection.
    """

    def __init__(self):
        """
        Initialize the pythonValidator with GPT client and prompts.

        Sets up:
        - GPT client with configuration parameters
        - Test generation prompt template
        - Python code detection prompt template
        """
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.generate_test_prompt = prompt.TESTCODE_GENERATION_PROMPT
        self.check_python_prompt = prompt.IS_PYTHON_PROMPT
        
    async def validate(self, task_obj, result, history) -> tuple[str, str]:
        """
        Validate Python code by generating and executing test code.

        This method:
        1. Generates test code based on task and result
        2. Executes the test code
        3. Logs validation results
        4. Determines validation status

        Args:
            task_obj: The task object containing validation requirements.
            result (str): The Python code to validate.

        Returns:
            tuple[str, str]: A tuple containing (error_message, status).
                           status can be 'completed' or 'failed'.
                           error_message is None if validation succeeds.
        """
        print('------Run pythonValidator.validate()------')

        test_code = await self.generate_test_function(task_obj, result, history)

        result_for_test = preprocessing.extract_functions(result)
        # and concat result_for_test with test_code  ....
        test_code = result_for_test + test_code
        runresult = await self.execute_python_code(test_code)

        # Logging execution result for debugging
        with open('validate_log.json', 'a', encoding='utf-8') as file:
            file.write('----------\nGOT PYTHON VALIDATION\n----------')
            json.dump({'task_obj': task_obj, 'result': result, 'testcode': test_code, 'runresult': runresult}, file, indent=4)

        print(f'***runresult is: {runresult}***')

        if 'Error executing code:' not in runresult:
            print('***Python code with no bugs***')
            return None, 'completed', test_code, runresult
        else:
            print('***Python code with bugs***')
            return runresult, 'failed', test_code, runresult
    
    async def generate_test_function(self, task_obj, result, history) -> str:
        """
        Generate test code for validating Python code execution.

        Args:
            task_obj: The task object containing test requirements.
            result (str): The Python code to test.
            history (str): The history of the task.

        Returns:
            str: Generated test code that validates the result.

        Note:
            The generated test code is stripped of Python markdown formatting.
        """
        print('------Run pythonValidator.generate_test_function()------')


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

        messages_exe = [
            {'role': 'system', 'content': self.generate_test_prompt},
            {'role': 'user', 'content': user_content}
        ]

        test_code = await self.gpt_client.a_chat_completion(messages_exe, temperature=Config.TEMPERATURE)
        test_code = test_code.strip('```python').strip('```')

        return test_code
    
    async def execute_python_code(self, test_code) -> str: 
        """
        Execute Python code safely and capture its output.

        This method:
        1. Redirects stdout to capture output
        2. Executes the code in a controlled environment
        3. Handles any execution errors
        4. Restores stdout

        Args:
            test_code (str): The Python code to execute.

        Returns:
            str: The execution output or error message.
        """
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
    
    async def is_python_code(self, result, task_obj) -> bool:
        """
        Check if the given result contains executable Python code.

        Args:
            result (str): The content to check for Python code.
            task_obj: The task object providing context for the check.

        Returns:
            bool: True if the content contains executable Python code, False otherwise.

        Note:
            This method uses GPT to analyze the content and determine if it's
            valid Python code that should be executed.
        """
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

