from pythonvalidator import pythonValidator
from textvalidator import textValidator

class Validator:
    def __init__(self):
        self.pythonval = pythonValidator()
        self.textval = textValidator()

    async def validate(self, task_obj, result, history):
        print('------Run Validator.validate()------')
        # judge whether the result contains python code
        
        
        if '```python' in result:
            is_python_code = await self.pythonval.is_python_code(result,task_obj)
            if is_python_code:
                # if the result contains python code
                return await self.pythonval.validate(task_obj, result)
        else:
            # if the result not contains python code
            return await self.textval.validate(task_obj, result, history)
        