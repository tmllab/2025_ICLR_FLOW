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
        print('---执行了execute_python_code---')
        try:
            print('---执行了一层---')
            # Create local namespace and capture output
            local_namespace = {}
            from io import StringIO
            import sys
            output = StringIO()
            sys.stdout = output
            
            try:
                print('执行了二层')
                # Execute code
                print('验证了完整代码')
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
                print('验证了类/函数')
                test_results = []
                for name, obj in defined_objects.items():
                    if isinstance(obj, type):  # Class
                        print('验证了类')
                        try:
                            instance = obj()
                            test_results.append(f"Created instance of class {name}")
                        except Exception as e:
                            test_results.append(f"Class {name} error: {str(e)}")
                    else:  # Function
                        print('验证了函数')
                        try:
                            # Try with no arguments first
                            result = obj()
                            test_results.append(f"Function {name} executed successfully")
                        except TypeError:
                            test_results.append(f"Function {name} requires parameters")
                        except Exception as e:
                            test_results.append(f"Function {name} error: {str(e)}")
                
                execution_output = "\n".join(test_results)
            
            # # Validate with GPT
            # validation_prompt = f"""
            # Task: {task_obj}
            # Code execution result: {execution_output}
            
            # Does this result correctly fulfill the task requirements? 
            # If yes, respond with 'CORRECT'. If no, explain why it's incorrect.
            # """
            
            # messages = [
            #     {'role': 'system', 'content': prompt.VALIDATION_PROMPT},
            #     {'role': 'user', 'content': validation_prompt}
            # ]
            
            # gpt_validation = await self.gpt_client.a_chat_completion(messages)
            
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