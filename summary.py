
from gptClient import GPTClient
from config import Config
import prompt

class Summary:
    """Run a subtask using GPT."""

    def __init__(self):

        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.summary_prompt = prompt.SUMMARY_PROMPT
    def summary(self, task, chathistory):
        # messages = [
        #     {'role': 'system', 'content': f'''
        #         You will be given the workflow for {task} in JSON format. Your job is to:
            
        #             1. Transform the JSON workflow into the required output format. Depending on the task, this could be:
        #                 Python code: Generate a .py file if the task is programming-related.
        #                 LaTeX file: Create a .tex file, such as a Beamer presentation, for documentation or slides.
        #                 Other formats as specified in the task.

        #             2. Review and integrate outputs from all subtasks in the workflow. Ensure the final output is comprehensive and not based solely on the result of the last subtask.

        #             3. Focus on producing the actual deliverable:
        #                 If the task specifies Python code, output a Python script.
        #                 If it asks for a LaTeX file, provide the full LaTeX document.
        #                 Avoid just summarizing the steps or describing the results—your primary goal is to create the actual output.

        #             4. Keep in mind:
        #                 Always generate the output in the format specified by the task.
        #                 Ensure the final result is complete, well-structured, and ready to use.
        #     '''},
        #     {'role': 'user', 'content': f'''
        #         Here's the workflow: {chathistory}
        #     '''}
        user_content = f'''
            Here is the task description: {task}
            Here is the workflow for the task: {chathistory}
        '''

        messages = [
            {'role': 'system', 'content': self.summary_prompt},
            {'role': 'user', 'content': user_content}
        ]

        return self.gpt_client.chat_completion(messages)

