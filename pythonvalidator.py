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


    async def check_windows(self, result) -> str:
        """
        check if
        """
       

        user_content = f"""
You are a precise and detail-oriented code-analysis assistant. Your task is to analyze the given Python code and extract **only** the lines that create or trigger new UI windows, dialogs, alerts, or interactive popups.

### **Instructions:**
1. Identify any lines that **actively display UI elements** such as:
   - **Message boxes, alerts, and dialogs** (e.g., `tkinter.messagebox.showinfo`, `showwarning`, `showerror`, `askquestion`, `askokcancel`, `askyesno`, `askretrycancel`)
   - **New windows that actively appear on the screen** (e.g., `tk.Toplevel()`, `QDialog.exec_()`, `wx.Dialog.ShowModal()`)
   - **Blocking UI elements** (e.g., `root.mainloop()`, `plt.show()`, `some_window.show()`, `some_dialog.exec_()`)

2. **Exclude** lines that merely **initialize** a window or object **without** displaying it, such as:
   - `root = tk.Tk()` (which only creates a window but does not start interaction)
   - `app = QApplication(sys.argv)` (which initializes the app but does not show a UI)
   - `figure = plt.figure()` (which creates but does not display a figure)

3. **Output only the exact lines of code that match the criteria.**  
   - Do **not** include any explanations, reasoning, or extra text.  
   - Do **not** reformat or modify the extracted lines.  
   - If multiple lines are detected, return them in the same order as they appear in the original code.

4. **If no such lines exist, output only an empty string (`""`).**

{result}
        """

        messages_exe = [
            {'role': 'user', 'content': user_content}
        ]

        popup = await self.gpt_client.a_chat_completion(messages_exe, temperature=Config.TEMPERATURE)
        popup = popup.strip('```python').strip('```')

        return popup
    



    def comment_out(self, original_code: str, llm_output: str) -> str:
        """
        Takes the original code and the LLM's output (lines to be commented out).
        Returns the modified code where the specified lines are commented.

        Parameters:
        - original_code (str): The full source code.
        - llm_output (str): The exact lines that need to be commented.

        Returns:
        - str: The modified code with the specified lines commented.
        """
        # Convert code into a list of lines
        code_lines = original_code.splitlines(keepends=True)

        # Convert LLM output into a set of stripped lines (to ignore whitespace mismatches)
        lines_to_comment = set(line.strip() for line in llm_output.splitlines() if line.strip())

        # Iterate through the code and comment out matching lines
        modified_lines = []
        for line in code_lines:
            stripped_line = line.strip()
            if stripped_line in lines_to_comment:
                modified_lines.append("# " + line)  # Prepend "#" while preserving indentation
            else:
                modified_lines.append(line)

        return "".join(modified_lines)


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
        result = result.strip('```python').strip('```')
        pops = await self.check_windows(result)
        code_for_test= self.comment_out(result, pops)
        test_funtions = await self.generate_test_function(task_obj, code_for_test, history)

        # and concat result_for_test with test_code  ....
        test_code = code_for_test + "\n"+test_funtions

        runresult = await self.execute_python_code(test_code)

        # Logging execution result for debugging
        with open('validate_log.json', 'a', encoding='utf-8') as file:
            file.write('----------\nGOT PYTHON VALIDATION\n----------')
            json.dump({'task_obj': task_obj, 'result': result, 'testcode': test_code, 'runresult': runresult}, file, indent=4)


        if 'Error executing code:' not in runresult:
            print('***Python code with no bugs***')
            return None, 'completed'
        else:
            print('***Python code with bugs***')
            return "###below are some unit tests: \n ```python"+ test_funtions+"``` ### tests results are as follows: \n"+ runresult, 'failed'
    
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
    
    async def need_validate(self, result, task_obj) -> bool:
        """
        """
        print('------Run pythonValidator.is_python_code()------')
        ##TODO check if the result need to be runned based on task_obj, 
        # it is possible that the code is just for explaination  
        user_content = f'''
You are an expert Python evaluator. Your task is to determine whether 
the provided Python code should be **unit tested** based on the task objective 
and its generated result.

### **Instructions:**
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

### **Task Objective:**
{task_obj}

    ----
### **Generated Python Code:**
{result}
        '''
        
        messages = [
            {'role': 'user', 'content': user_content}
        ]
     
        
        feedback = await self.gpt_client.a_chat_completion(messages, temperature=Config.TEMPERATURE)
       
        
        if feedback == "N":
            return False
        else:
            return True

