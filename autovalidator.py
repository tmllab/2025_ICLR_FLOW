from pythonvalidator import pythonValidator
from textvalidator import textValidator

class Validator:
    """
    A high-level validator class that orchestrates both Python and text validation.

    This class serves as a facade for different types of validation:
    - Determines whether content contains Python code
    - Routes validation to appropriate validator (Python or text)
    - Manages validation results and feedback

    Attributes:
        pythonval (pythonValidator): Validator for Python code execution.
        textval (textValidator): Validator for text-based content.
    """

    def __init__(self):
        """
        Initialize the Validator with specialized validators.

        Sets up:
        - Python code validator for code execution validation
        - Text validator for non-code content validation
        """
        self.pythonval = pythonValidator()
        self.textval = textValidator()

    async def validate(self, task_obj, solution, history) -> tuple[str, str]:
        """
        Validate task results by determining content type and using appropriate validator.

        This method:
        1. Checks if the result contains Python code
        2. Routes to Python validator if code is detected
        3. Routes to text validator for non-code content
        4. Returns validation results and status

        Args:
            task_obj: The task object containing validation requirements.
            result (str): The content to validate (could be code or text).
            history (str): Previous execution history and feedback.

        Returns:
            tuple[str, str]: A tuple containing (feedback, status).
                           status can be 'completed' or 'failed'.
                           feedback is None if validation succeeds,
                           otherwise contains improvement suggestions.

        Note:
            The method first checks for Python code markers ('```python')
            and then confirms if the content is actually executable Python code
            before routing to the appropriate validator.
        """
        # judge whether the result contains python code
        temp = solution.lstrip()
        import re
 
    

        is_python_code = await self.pythonval.need_validate(solution,task_obj)
        if is_python_code:
            # if the result contains python code
            test_result, status = await self.pythonval.validate(task_obj, solution, history)

            
        else:
            test_result, status =  await self.textval.validate(task_obj, solution, history)

        # unit_test_result = unit_test_result+text_validate_result
        
        return test_result, status
        