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
        执行Python代码并验证结果是否符合任务要求
        
        Args:
            task_obj: 任务描述
            result: 要执行的Python代码
        
        Returns:
            None: 如果代码执行成功且符合要求
            str: 如果有错误或不符合要求，返回反馈信息
        """
        print('------EXECUTE AND VALIDATE PYTHON CODE------')
        
        try:
            # 创建一个本地命名空间来执行代码
            local_namespace = {}
            
            # 捕获标准输出
            from io import StringIO
            import sys
            output = StringIO()
            sys.stdout = output
            
            try:
                # 执行代码
                exec(result, {}, local_namespace)
                print('执行了')
                # 获取打印输出
                printed_output = output.getvalue()
                printed_output = '测试结果有输出'
                print(output)
                print(printed_output)
                
                # 检查是否定义了函数或类
                defined_objects = {
                    name: obj for name, obj in local_namespace.items()
                    if callable(obj) and not name.startswith('__')
                }
                
                execution_result = {
                    'success': True,
                    'output': printed_output,
                    'namespace': local_namespace,
                    'defined_objects': defined_objects
                }
                
            finally:
                # 恢复标准输出
                sys.stdout = sys.__stdout__
            
            # 验证执行结果是否符合任务要求
            if defined_objects:
                print(defined_objects)
                # 如果定义了函数或类，生成测试用例
                test_results = []
                for name, obj in defined_objects.items():
                    if isinstance(obj, type):  # 类
                        print('测试了类')
                        try:
                            instance = obj()
                            test_results.append(f"Successfully created instance of {name}")
                        except Exception as e:
                            return f"Class {name} initialization failed: {str(e)}"
                    else:  # 函数
                        print('测试了函数')
                        try:
                            # 尝试基本调用（这里可以根据任务要求添加具体的测试用例）
                            result = obj()
                            test_results.append(f"Function {name} executed successfully")
                        except TypeError:  # 如果需要参数
                            test_results.append(f"Function {name} requires parameters")
                        except Exception as e:
                            return f"Function {name} execution failed: {str(e)}"
                print('测试函数和类了')
                if test_results:
                    return test_results
                else:
                    print('No valid test results')
                    return 'No valid test results'
                
                
            else:
                # 如果是直接执行的代码，检查是否有输出
                if printed_output.strip():
                    print(printed_output)
                    print('测试直接代码了成功且有输出')
                    return None  # 执行成功且有输出
                else:
                    return "Code executed but produced no output"
                    
        except Exception as e:
            return f"Code execution failed: {str(e)}"
        