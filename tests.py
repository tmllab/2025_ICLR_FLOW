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

# 测试例子：图形化绘图应用
task5 = "创建一个简单的绘图应用，包含画笔工具、形状工具和颜色选择器"
code5 = """
import tkinter as tk
from tkinter import colorchooser, ttk

class DrawingTool:
    def __init__(self, name, draw_function):
        self.name = name
        self.draw_function = draw_function

class Shape:
    def __init__(self, start_x, start_y, end_x, end_y, color):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.color = color

class Rectangle(Shape):
    def draw(self, canvas):
        return canvas.create_rectangle(
            self.start_x, self.start_y,
            self.end_x, self.end_y,
            fill=self.color
        )

class Circle(Shape):
    def draw(self, canvas):
        return canvas.create_oval(
            self.start_x, self.start_y,
            self.end_x, self.end_y,
            fill=self.color
        )

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("简单绘图应用")
        
        # 初始化变量
        self.current_tool = None
        self.current_color = "#000000"
        self.start_x = None
        self.start_y = None
        self.shapes = []
        
        self.setup_ui()
        self.setup_tools()
        
    def setup_ui(self):
        # 创建工具栏
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # 创建画布
        self.canvas = tk.Canvas(
            self.root,
            width=600,
            height=400,
            bg="white"
        )
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # 绑定事件
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
    def setup_tools(self):
        # 创建工具按钮
        ttk.Button(
            self.toolbar,
            text="矩形",
            command=lambda: self.select_tool("rectangle")
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            self.toolbar,
            text="圆形",
            command=lambda: self.select_tool("circle")
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            self.toolbar,
            text="选择颜色",
            command=self.choose_color
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            self.toolbar,
            text="清除",
            command=self.clear_canvas
        ).pack(side=tk.LEFT, padx=5)
        
    def select_tool(self, tool_name):
        self.current_tool = tool_name
        
    def choose_color(self):
        color = colorchooser.askcolor(color=self.current_color)[1]
        if color:
            self.current_color = color
            
    def clear_canvas(self):
        self.canvas.delete("all")
        self.shapes = []
        
    def start_drawing(self, event):
        self.start_x = event.x
        self.start_y = event.y
        
    def draw(self, event):
        if not self.current_tool:
            return
            
        # 删除上一次的预览
        if hasattr(self, 'preview_shape'):
            self.canvas.delete(self.preview_shape)
            
        # 创建新的预览
        if self.current_tool == "rectangle":
            shape = Rectangle(
                self.start_x, self.start_y,
                event.x, event.y,
                self.current_color
            )
        else:  # circle
            shape = Circle(
                self.start_x, self.start_y,
                event.x, event.y,
                self.current_color
            )
            
        self.preview_shape = shape.draw(self.canvas)
        
    def stop_drawing(self, event):
        if not self.current_tool:
            return
            
        # 创建最终形状
        if self.current_tool == "rectangle":
            shape = Rectangle(
                self.start_x, self.start_y,
                event.x, event.y,
                self.current_color
            )
        else:  # circle
            shape = Circle(
                self.start_x, self.start_y,
                event.x, event.y,
                self.current_color
            )
            
        self.shapes.append(shape)
        shape.draw(self.canvas)
        
        if hasattr(self, 'preview_shape'):
            self.canvas.delete(self.preview_shape)
            del self.preview_shape

# 创建并运行应用
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
"""

def run_tests():
    tests = [
        (task1, code1),
        (task2, code2),
        (task3, code3),
        (task4, code4),
        (task5, code5)  # 添加新的测试用例
    ]
    
    for task, code in tests:
        print(f"\n测试任务: {task}")
        print("代码:", code)
        result = execute_python_code(task, code)
        print("执行结果:", result)

run_tests()
