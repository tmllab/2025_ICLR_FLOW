from gptClient import GPTClient
from config import Config
import prompt
from logging_config import get_logger, log_gpt_conversation

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
    
    async def execute(self, subtask: str, agent_id: str, context: str, next_objective: str, output_format: str = '') -> str:
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
            output_format (str): Required output format for this task.

        Returns:
            str: The execution result from the GPT model.
        """
        logger = get_logger('execution')
        logger.info(f"Task '{subtask}' started by agent '{agent_id}'.")

        user_content_parts = [
            f"# **The Overall Goal**\n{self.overall_task}\n\n---\n",
            f"# **Context from Upstream Tasks**\n{context}\n\n---\n",
            f"# **Downstream Tasks Objectives**\n{next_objective}\n\n---\n",
            f"# **Current Task Requirements**\n{subtask}"
        ]
        
        # Add output format requirement if specified
        if output_format.strip():
            user_content_parts.append(f"\n\n---\n\n# **Required Output Format**\n{output_format}")
        
        user_content = "".join(user_content_parts)

        messages = [
            { "role": "system", "content": self.execute_prompt },
            { "role": "user", "content": user_content }
        ]

        logger.info("Sending execution request to GPT")
        logger.debug(f"GPT messages for task execution: {len(messages)} messages")

        result = await self.gpt_client.a_chat_completion(messages)

        # Log the complete GPT conversation for debugging
        log_gpt_conversation(
            task_id=f"task_{agent_id}", 
            conversation_type="task_execution",
            messages=messages,
            response=result,
            iteration=1
        )

        logger.info(f"Task execution completed, result length: {len(result)}")
        return result

    async def re_execute(self, subtask: str, context: str, next_objective: str, result: str, history_messages: list, output_format: str = '') -> str:
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
        user_content_parts = [
            f"# **The Overall Goal**\n{self.overall_task}\n\n---\n",
            f"# **Context from Upstream Tasks**\n{context}\n\n---\n",
            f"# **Downstream Tasks Objectives**\n{next_objective}\n\n---\n",
            f"# **Current Task Requirements**\n{subtask}"
        ]
        
        # Add output format requirement if specified
        if output_format.strip():
            user_content_parts.append(f"\n\n---\n\n# **Required Output Format**\n{output_format}")
        
        user_message = {
            "role": "user",
            "content": "".join(user_content_parts)
        }


        # Step 4: Combine into one message list
        messages = [system_message] + history_messages + [user_message]

        logger = get_logger('execution')
        logger.info("Sending re-execution request to GPT")
        logger.debug(f"GPT re-execution messages: {len(messages)} messages (including history)")

        # Step 5: Send to GPT
        result = await self.gpt_client.a_chat_completion(messages)

        # Log the complete GPT re-execution conversation
        log_gpt_conversation(
            task_id=f"task_{subtask[:20]}",  # Use first 20 chars of subtask as ID
            conversation_type="task_re_execution",
            messages=messages,
            response=result,
            iteration=len(history_messages) // 2 + 1  # Estimate iteration from history
        )

        logger.info(f"Task re-execution completed, result length: {len(result)}")
        return result