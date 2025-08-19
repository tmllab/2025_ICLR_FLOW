import asyncio
from gptClient import GPTClient
from config import Config
import json
from code_test_module import CodeTester

class pythonValidator:
    """
    A validator class that checks Python code execution and validity.

    This class is responsible for:
    - Validating Python code execution using intelligent testing strategies
    - Determining if code needs validation based on task context
    - Integrating with the code test module for appropriate testing

    Attributes:
        gpt_client (GPTClient): Client for interacting with GPT API.
        code_tester (CodeTester): Intelligent code testing module.
    """

    def __init__(self):
        """
        Initialize the pythonValidator with GPT client and code tester.

        Sets up:
        - GPT client with configuration parameters
        - Code testing module for intelligent testing
        """
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.code_tester = CodeTester()

    async def validate(self, task_obj, result, history) -> tuple[str, str]:
        """
        Validate Python code using intelligent testing strategies.

        This method:
        1. Uses the code test module to determine appropriate testing strategy
        2. Executes validation based on code complexity and type
        3. Logs validation results
        4. Returns validation status

        Args:
            task_obj: The task object containing validation requirements.
            result (str): The Python code to validate.
            history: Previous execution history.

        Returns:
            tuple[str, str]: A tuple containing (feedback, status).
                           status can be 'completed' or 'failed'.
        """
        try:
            print("********Starting intelligent Python validation************")
            
            # Clean the code
            code_for_test = result.strip('```python').strip('```').strip()
            
            if not code_for_test:
                return "No Python code found to validate", 'failed'
            
            # Use the intelligent code tester
            test_result, status = await self.code_tester.test_code(code_for_test, str(task_obj))
            
            # Log validation results
            with open('validate_log.json', 'a', encoding='utf-8') as file:
                file.write('\n----------\nINTELLIGENT PYTHON VALIDATION\n----------\n')
                log_data = {
                    'task_obj': str(task_obj),
                    'result': code_for_test[:500] + '...' if len(code_for_test) > 500 else code_for_test,
                    'test_result': test_result,
                    'status': status
                }
                json.dump(log_data, file, indent=2, ensure_ascii=False)
                file.write('\n')
            
            if status == 'completed':
                print('***Python code validation passed***')
                return f"### **Code Validation Results:**\n{test_result}", 'completed'
            else:
                print('***Python code validation failed***')
                return f"### **Code Validation Issues:**\n{test_result}", 'failed'
                
        except Exception as e:
            error_msg = f"Validation system error: {str(e)}"
            print(f"***{error_msg}***")
            return error_msg, 'failed'
    
    async def need_validate(self, result, task_obj) -> bool:
        """
        Determine whether the provided Python code should be validated based on task objective.
        
        Args:
            result (str): The Python code result
            task_obj (str): The task objective
            
        Returns:
            bool: True if code should be validated, False otherwise
        """
        print('------Run pythonValidator.need_validate()------')
        
        user_content = f'''
Determine whether the provided Python code should be **unit tested** based on the task objective and its generated result.

# **Instructions:**
- If the task objective specifies generating a **program or executable logic** for future use, 
and the result contains **functional Python code (e.g., functions, classes, or logic meant for execution)**,  
**return `Y`**.
- If the task objective is **purely educational, explanatory, or illustrative**, 
and the result contains a **non-functional snippet meant only for explanation**,  
**return `N`**.
- If the result contains **incomplete code, pseudocode, or non-executable elements**,  
**return `N`**.
- **Do not include any explanations or additional text.**  
Your response must be strictly either `Y` or `N`.

# **Task Objective:**
{task_obj}

----
# **Generated Python Code:**
{result}
        '''
        
        messages = [
            {'role': 'user', 'content': user_content}
        ]
        
        feedback = await self.gpt_client.a_chat_completion(messages)
        
        if feedback == "N":
            return False
        else:
            return True