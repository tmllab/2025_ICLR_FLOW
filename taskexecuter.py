from gptClient import GPTClient
from config import Config
import prompt
import logging

# -----------------------------------------------------------------------------
# Configuration and Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Task Execution
# -----------------------------------------------------------------------------
class taskExecuter:
    """
    Handles the execution of individual tasks using GPT-based language models.

    This class is responsible for:
    - Managing GPT client configuration
    - Executing tasks with appropriate context
    - Handling task re-execution based on feedback
    - Formatting prompts for task execution

    Attributes:
        gpt_client (GPTClient): Client for interacting with GPT API.
        overall_task (str): The main task description providing global context.
        execute_prompt (str): Template prompt for initial task execution.
        re_execute_prompt (str): Template prompt for task re-execution.
    """
    def __init__(self, overall_task):
        """
        Initialize the taskExecuter with necessary configurations.

        Args:
            overall_task (str): The main task description that provides context
                              for all subtask executions.

        Note:
            Initializes GPT client with configuration parameters from Config class
            and sets up execution prompts.
        """
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.overall_task = overall_task
        self.execute_prompt = prompt.TASK_EXECUTION_PROMPT
        self.re_execute_prompt = prompt.TASK_REEXECUTION_PROMPT
    
    async def execute(self, subtask: str, agent_id: str, context: str, next_objective: str) -> str:
        """
        Execute a task for the first time to get initial results.

        This method constructs a comprehensive prompt that includes:
        - Context from previously completed tasks
        - Overall goal of the workflow
        - Specific subtask to be executed
        - Objectives of subsequent tasks

        Args:
            subtask (str): The specific task to be executed.
            agent_id (str): Identifier for the agent executing the task.
            context (str): Context from previously completed tasks.
            next_objective (str): Objectives of downstream tasks.

        Returns:
            str: The execution result from the GPT model.

        Note:
            The method formats the prompt to ensure the model:
            1. Focuses only on the assigned subtask
            2. Uses context appropriately
            3. Aligns with overall goals
            4. Provides output suitable for downstream tasks
        """
        logger.info(f"Task '{subtask}' started by agent '{agent_id}'.")
        
        # User prompt with context and objectives
        user_content = (
            f"### Context from Completed Tasks:\n{context}\n\n"
            f"### Overall Goal:\n{self.overall_task}\n\n"
            f"### Your Subtask:\n{subtask}\n\n"
            f"### Subsequent Task Objectives:\n{next_objective}\n\n"
            "Instructions:\n"
            "1. Solve only your assigned subtask, referring to the context only if necessary.\n"
            "2. Ensure your solution aligns with the overall goal and is formatted so that it can be directly used as input for downstream tasks.\n"
            "3. Do not repeat any previous output verbatim.\n"
        )

        messages = [
            {'role': 'system', 'content': self.execute_prompt},
            {'role': 'user', 'content': user_content}
        ]

        result = await self.gpt_client.a_chat_completion(messages, temperature=Config.TEMPERATURE)

        return result

    async def re_execute(self, subtask: str, context: str, next_objective: str, result: str, history: str) -> str:
        """
        Re-execute a task based on previous results and feedback.

        This method is called when initial execution needs improvement. It provides:
        - Previous execution history
        - Original context and objectives
        - Clear instructions for refinement

        Args:
            subtask (str): The specific task to be refined.
            context (str): Context from parent tasks.
            next_objective (str): Objectives of child tasks.
            result (str): Previous execution result.
            history (str): History of previous execution attempts and feedback.

        Returns:
            str: The refined execution result from the GPT model.

        Note:
            This method is specifically designed for iterative improvement,
            taking into account previous attempts and feedback to produce
            better results.
        """
        user_content = f"""
            ## You need to further refine the subtask results based on following information


            ## Context from Parent Tasks:
            {context}

            ---

            ##Child Tasks objectives:
            {next_objective}

            ---

            ## The Overall Goal:
            {self.overall_task}

            ---

            ## current task Requirement:
            {subtask}

            ---

            ## current task Change History:
            {history}

        """
        

        messages = [
            {'role': 'system', 'content': self.re_execute_prompt},
            {'role': 'user', 'content': user_content}
        ]
        print(user_content)
        result = await self.gpt_client.a_chat_completion(messages, temperature=Config.TEMPERATURE)

        if result:
            print('------Re-execute completed------')
        return result