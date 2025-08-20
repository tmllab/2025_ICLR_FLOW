import asyncio
from gptClient import GPTClient
from config import Config
from code_test_module import CodeTester
from logging_config import get_logger, log_validation_result

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
        self.logger = get_logger('validation')

    async def validate(self, task_obj, result, history, overall_task: str = None, output_format: str = None) -> tuple[str, str]:
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
            self.logger.info("Starting intelligent Python code validation")
            
            # Clean the code
            code_for_test = result.strip('```python').strip('```').strip()
            
            if not code_for_test:
                self.logger.warning("No Python code found to validate")
                return "No Python code found to validate", 'failed'
            
            # Use the intelligent code tester
            test_result, status = await self.code_tester.test_code(code_for_test, str(task_obj))
            
            # Log validation results using structured logging
            task_id = getattr(history, 'task_id', 'unknown') if hasattr(history, 'task_id') else 'unknown'
            log_validation_result(
                task_id=task_id,
                task_obj=str(task_obj),
                result=code_for_test,
                validation_type='python_intelligent',
                status=status,
                feedback=test_result
            )
            
            if status == 'completed':
                self.logger.info("Python code validation passed")
                return f"### **Code Validation Results:**\n{test_result}", 'completed'
            else:
                self.logger.warning(f"Python code validation failed: {test_result[:100]}...")
                return f"### **Code Validation Issues:**\n{test_result}", 'failed'
                
        except Exception as e:
            error_msg = f"Validation system error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
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
        self.logger.debug(f"Determining if Python code validation is needed for task: {task_obj[:50]}...")
        
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