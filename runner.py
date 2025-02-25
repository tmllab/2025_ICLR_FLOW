from gptClient import GPTClient
import logging
from config import Config
from workflow import Workflow
from prompt import RUNNER_PROMPT

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
# Async Runner for Task Execution
# -----------------------------------------------------------------------------
class AsyncRunner:
    """Executes an individual task asynchronously via GPT."""
    def __init__(self, overall_task: str):
        self.overall_task = overall_task
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )

    async def _execute(self, subtask: str, agent_id: str, context: str, next_objective: str) -> str:
        logger.info(f"Task '{subtask}' started by agent '{agent_id}'.")
        
        # System instructions for GPT
        system_content = RUNNER_PROMPT

        print(f'object: {subtask}')
        print(f'next: {next_objective}')
        
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
            "4. Avoid contractions in your response.\n"
        )

        messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': user_content}
        ]

        result = await self.gpt_client.a_chat_completion(messages, temperature=Config.TEMPERATURE)
        return result

    async def execute(self, workflow: Workflow, task_id: str) -> str:
        """Wraps task execution logic, incorporating context and downstream objectives."""
        if task_id not in workflow.tasks:
            logger.error(f"Task '{task_id}' not found in workflow.")
            return f"Error: Task '{task_id}' not found."
        
        task_obj = workflow.tasks[task_id]
        # Get context from completed previous tasks as a nicely formatted string.
        context = workflow.get_context(task_id)
        task_objective = task_obj.objective
        # Get downstream tasks objectives as a nicely formatted string.
        next_objective = workflow.get_downsteam_objectives(task_id)
        agent_id = task_obj.agent_id
        
        # Log a brief snippet of the context (first 100 characters) for clarity.
        logger.info(f"Executing task '{task_objective}' with context: {context[:100]}...")
        result = await self._execute(task_objective, agent_id, context, next_objective)
        return result