from gptClient import GPTClient
from config import Config
import prompt

class Summary:
    """
    A class that generates summaries of task execution results using GPT.

    This class is responsible for:
    - Initializing GPT client for summary generation
    - Processing task descriptions and chat histories
    - Generating concise summaries of workflow execution

    Attributes:
        gpt_client (GPTClient): Client for interacting with GPT API.
        summary_prompt (str): Template prompt for generating summaries.
    """

    def __init__(self):
        """
        Initialize the Summary class with GPT client configuration.

        Sets up:
        - GPT client with appropriate API key, model, and temperature
        - Summary generation prompt template

        Note:
            Uses configuration parameters from Config class for GPT settings.
        """
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model='gpt-4o-mini',
            temperature=Config.TEMPERATURE
        )
        self.summary_prompt = prompt.RESULT_EXTRACT_PROMPT
        
    def summary(self, task: str, chathistory: str) -> str:
        """
        Generate a summary of task execution based on chat history.

        This method creates a summary by:
        1. Combining task description with workflow history
        2. Using GPT to extract key information
        3. Generating a concise summary of the execution

        Args:
            task (str): Description of the task that was executed.
            chathistory (str): Complete history of the workflow execution.

        Returns:
            str: A concise summary of the task execution results.

        Note:
            The summary focuses on extracting the most relevant information
            from the chat history in the context of the given task.
        """
        user_content = f'''
            Here is the task description: {task}
            Here is the workflow for the task: {chathistory}
        '''

        messages = [
            {'role': 'system', 'content': self.summary_prompt},
            {'role': 'user', 'content': user_content}
        ]

        return self.gpt_client.chat_completion(messages)

