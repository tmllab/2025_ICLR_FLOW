from taskexecuter import taskExecuter
import logging
from config import Config
from workflow import Workflow
from autovalidator import Validator
from workflow import Task
import prompt
 
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
    """
    Executes individual tasks asynchronously using GPT-based task execution.

    This class handles the execution of tasks, including validation and re-execution
    if necessary. It maintains the execution context and manages the interaction
    between task execution and validation.

    Attributes:
        overall_task (str): The main task description for the entire workflow.
        max_validation_itt (int): Maximum number of validation/re-execution iterations.
        executer (taskExecuter): Instance handling the actual task execution.
        validator (Validator): Instance handling result validation.
    """
    def __init__(self, overall_task: str, max_validation_itt):
        """
        Initialize the AsyncRunner with task execution parameters.

        Args:
            overall_task (str): The main task description that provides context for execution.
            max_validation_itt (int): Maximum number of validation iterations allowed per task.
        """
        self.overall_task = overall_task
        self.max_validation_itt = max_validation_itt
        self.executer = taskExecuter(overall_task)
        self.validator = Validator()

 
    async def execute(self, workflow: Workflow, task_id: str) -> str:
        """
        Executes a task within a workflow, handling validation and potential re-execution.

        This method:
        1. Validates task existence in workflow
        2. Retrieves necessary context from previous tasks
        3. Gets downstream objectives
        4. Executes the task and validates the result
        5. Re-executes if necessary (up to max_validation_itt times)
        6. Stores execution history and logs results

        Args:
            workflow (Workflow): The workflow containing the task and its relationships.
            task_id (str): Unique identifier of the task to execute.

        Returns:
            str: The final execution result, whether perfect or after max iterations.

        Raises:
            ValueError: If the task_id is not found in the workflow.
        """

        
        if task_id not in workflow.tasks:
            logger.error(f"Task '{task_id}' not found in workflow.")
            return f"Error: Task '{task_id}' not found."
        
        task_obj = workflow.tasks[task_id]
        context = workflow.get_context(task_id)
        task_objective = task_obj.objective
        next_objective = workflow.get_downsteam_objectives(task_id)
        agent_id = task_obj.agent_id
        
        logger.info(f"Executing task '{task_objective}' with context: {context[:100]}...")
        

        result = await self.executer.execute(task_objective, agent_id, context, next_objective)
        
        # If validation is disabled, mark as completed immediately
        if self.max_validation_itt == 0:
            task_obj.save_history(result, '')
            task_obj.set_status("completed")
            return result

        # Validation loop with re-execution on failure
        for iteration in range(self.max_validation_itt):
            logger.info(f"Validation iteration {iteration + 1}/{self.max_validation_itt} for task '{task_id}'")
            
            # Validate the current result
            feedback, new_status = await self.validator.validate(task_obj.objective, result, task_obj.get_history())
            
            if new_status == 'completed':
                logger.info(f"Task '{task_id}' validated successfully after {iteration + 1} iteration(s)")
                task_obj.save_history(result, feedback)
                task_obj.set_status("completed")
                return result
            
            # Validation failed - save the failed attempt
            task_obj.save_history(result, feedback)
            logger.info(f"Validation failed for task '{task_id}': {feedback[:100]}...")
            
            # Re-execute for the next validation attempt (except on last iteration)
            if iteration < self.max_validation_itt - 1:
                logger.info(f"Re-executing task '{task_id}' for attempt {iteration + 2}")
                result = await self.executer.re_execute(task_objective, context, next_objective, result, task_obj.get_history())

        # If we exit the loop, all validation attempts failed
        logger.warning(f"Task '{task_id}' failed validation after {self.max_validation_itt} attempts. Marking as failed.")
        task_obj.set_status("failed")
        return result
