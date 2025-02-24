def execute_python_code(task_obj, result):
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
    try:
        # Create local namespace and capture output
        local_namespace = {}
        from io import StringIO
        import sys
        output = StringIO()
        sys.stdout = output
        
        try:
            # Execute code
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
            test_results = []
            for name, obj in defined_objects.items():
                if isinstance(obj, type):  # Class
                    try:
                        instance = obj()
                        test_results.append(f"Created instance of class {name}")
                    except Exception as e:
                        test_results.append(f"Class {name} error: {str(e)}")
                else:  # Function
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
    

# 测试例子1：简单函数
task1 = "创建一个返回'Hello World'的函数"
code1 = """
def say_hello():
    return "Hello World"
"""

# 测试例子2：带参数的函数
task2 = "创建一个计算两个数之和的函数"
code2 = """
def add_numbers(a, b):
    return a + b
"""

# 测试例子3：简单类
task3 = "创建一个Person类，包含name属性"
code3 = """
class Person:
    def __init__(self):
        self.name = "John"
"""

# 测试例子4：有语法错误的代码
task4 = "创建一个函数"
code4 = """
def broken_function()
    print("This has syntax error")
"""

# 运行测试
def run_tests():
    tests = [
        (task1, code1),
        (task2, code2),
        (task3, code3),
        (task4, code4)
    ]
    
    for task, code in tests:
        print(f"\n测试任务: {task}")
        print("代码:", code)
        result = execute_python_code(task, code)
        print("执行结果:", result)

run_tests()
