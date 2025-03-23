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

        Constructs a structured chat prompt with:
        - A system prompt (general behavior and formatting instructions)
        - A user message (task-specific context and goals)

        Args:
            subtask (str): The specific task to be executed.
            agent_id (str): Identifier for the agent executing the task.
            context (str): Context from previously completed tasks.
            next_objective (str): Objectives of downstream tasks.

        Returns:
            str: The execution result from the GPT model.
        """
        logger.info(f"Task '{subtask}' started by agent '{agent_id}'.")

        user_content = f"""
# **The Overall Goal**
{self.overall_task}

---

# **Context from Upstream Tasks**
{context}

---

# **Downstream Tasks Objectives**
{next_objective}

---

# **Current Task Requirements**
{subtask}
    """

        messages = [
            { "role": "system", "content": self.execute_prompt },
            { "role": "user", "content": user_content }
        ]

        print("*********** Execution Prompt *************")
        for m in messages:
            print(f"{m['role'].upper()}:\n{m['content']}\n")
        print("*********** Execution Prompt End *********")

        result = await self.gpt_client.a_chat_completion(messages)

        return result

    async def re_execute(self, subtask: str, context: str, next_objective: str, result: str, history_messages: list) -> str:
        """
        Re-execute a task based on previous results and feedback.

        This method uses structured chat history (assistant + user turns)
        from previous attempts for iterative refinement.
        """
        # Step 1: Build the system prompt
        system_message = {
            "role": "system",
            "content": self.re_execute_prompt  # your base instructions for GPT behavior
        }

        # Step 2: Instruction + current task info as the new user prompt
        user_message = {
            "role": "user",
            "content": f"""
# **The Overall Goal**
{self.overall_task}

---

# **Context from Upstream Tasks**
{context}

---

# **Downstream Tasks Objectives**
{next_objective}

---

# **Current Task Requirements**
{subtask}
"""
        }


        # Step 4: Combine into one message list
        messages = [system_message] + history_messages + [user_message]

        # Optional: Print for debugging
        print("***********reexecuter start*********************")
        for m in messages:
            print(f"{m['role'].upper()}: {m['content']}\n")
        print("***********reexecuter end*********************")

        # Step 5: Send to GPT
        result = await self.gpt_client.a_chat_completion(messages)

        return result