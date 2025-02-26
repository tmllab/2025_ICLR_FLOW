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
    """Executes an individual task asynchronously via GPT."""
    def __init__(self, overall_task):
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.overall_task = overall_task
        self.execute_prompt = prompt.EXECUTE_PROMPT
        self.re_execute_prompt = prompt.RE_EXECUTE_PROMPT
    
    async def execute(self, subtask: str, agent_id: str, context: str, next_objective: str):
        '''Execute the task for the first result'''

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

    async def re_execute(self, subtask: str, agent_id: str, context: str, next_objective: str, feedback: str):
        '''Re-execute the task following the feedback and result'''

        user_content = f'''
            Here is the subtask: {subtask}
            Here is the result: {result}
            Here is the validation feedback: {feedback}
        '''
        messages = [
            {'role': 'system', 'content': self.re_execute_prompt},
            {'role': 'user', 'content': user_content}
        ]

        result = await self.gpt_client.a_chat_completion(messages, temperature=Config.TEMPERATURE)

        return result