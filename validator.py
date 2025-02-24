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
    
    async def execute_python_code(self, task_obj, result):
        """
        Execute Python code and validate if it meets the task requirements
        
        Args:
            task_obj: Task description
            result: Python code to execute
        
        Returns:
            dict: Execution results containing:
                - success (bool): Whether execution succeeded
                - output (str): Execution output or error message
                - validation (str): GPT validation feedback
        """
        print('---execute_python_code---')
        try:
            print('---first layer--')
            # Create local namespace and capture output
            local_namespace = {}
            from io import StringIO
            import sys
            output = StringIO()
            sys.stdout = output
            
            try:
                print('second layer')
                # Execute code
                print('Whole code validated')
                exec(result, {}, local_namespace)
                execution_output = output.getvalue()
                
                # Get defined callable objects
                defined_objects = {
                    name: obj for name, obj in local_namespace.items()
                    if callable(obj) and not name.startswith('__')
                }
                
            finally:
                sys.stdout = sys.__stdout__

            # Handle execution results
            if defined_objects:
                # Test functions and classes
                print('Class/function validated')
                test_results = []
                for name, obj in defined_objects.items():
                    if isinstance(obj, type):  # Class
                        print('Class validated')
                        try:
                            instance = obj()
                            test_results.append(f"Created instance of class {name}")
                        except Exception as e:
                            test_results.append(f"Class {name} error: {str(e)}")
                    else:  # Function
                        print('Function validated')
                        try:
                            # Try with no arguments first
                            result = obj()
                            test_results.append(f"Function {name} executed successfully")
                        except TypeError:
                            test_results.append(f"Function {name} requires parameters")
                        except Exception as e:
                            test_results.append(f"Function {name} error: {str(e)}")
                
                execution_output = "\n".join(test_results)
            
            return {
                'success': True,
                'output': execution_output,
                #'validation': gpt_validation
            }
                
        except Exception as e:
            return {
                'success': False,
                'output': f"Execution error: {str(e)}",
                #'validation': "Code execution failed"
            }