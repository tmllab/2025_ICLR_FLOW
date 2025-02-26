from pythonvalidator import pythonValidator
from textvalidator import textValidator

class Validator:
    def __init__(self):
        self.pythonval = pythonValidator()
        self.textval = textValidator()

    async def validate(self, task_obj, result):
        print('------Run Validator.validate()------')
        # judge whether the result contains python code
        if await self.pythonval.is_python_code(result):
            # if the result contains python code
            return await self.pythonval.validate(task_obj, result)
        else:
            # if the result not contains python code
            return await self.textval.validate(task_obj, result)
        