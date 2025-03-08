from taskexecuter import taskExecuter
import logging
from config import Config
from workflow import Workflow
from validator import Validator
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

    async def _execute(self, task_obj: Task, subtask: str, agent_id: str, context: str, next_objective: str) -> str:
        """
        Internal method to execute a task with validation and potential re-execution.

        This method handles the core execution loop including:
        1. Initial task execution
        2. Result validation
        3. Re-execution if necessary (up to max_validation_itt times)
        4. History tracking of execution attempts

        Args:
            task_obj (Task): The task object being executed.
            subtask (str): The specific objective/instruction for this task.
            agent_id (str): Identifier for the agent executing the task.
            context (str): Context from previous task results.
            next_objective (str): Objectives of downstream tasks.

        Returns:
            str: The final execution result, whether perfect or after max iterations.

        Note:
            The method will attempt to improve results through multiple iterations
            if the validator indicates the result is not perfect.
        """
        print('------Run _execute------')
        i = 0
        result = ''

        while i < self.max_validation_itt:
            print(f'------Go into while loop, validating------ \ntask: {task_obj.id} \ntimes: {i}')
            if i == 0:
                result = await self.executer.execute(subtask, agent_id, context, next_objective)
            else:
                # re-execute here
                result = await self.executer.re_execute(subtask, context, next_objective, result, task_obj.get_history())

            feedback, new_status = await self.validator.validate(subtask, result,  task_obj.get_history())
            task_obj.save_history(result, feedback)
            task_obj.set_status(new_status)
            if new_status == 'completed' :
                print('---Result is perfect---')
                break
            i += 1
        print(task_obj.id, '---status is--',task_obj.status)
        return result

    async def execute(self, workflow: Workflow, task_id: str) -> str:
        """
        Main execution method that orchestrates the task execution process.

        This method:
        1. Validates task existence in workflow
        2. Retrieves necessary context from previous tasks
        3. Gets downstream objectives
        4. Triggers the actual execution process
        5. Handles result storage and logging

        Args:
            workflow (Workflow): The workflow containing the task and its relationships.
            task_id (str): Unique identifier of the task to execute.

        Returns:
            str: The execution result.

        Raises:
            ValueError: If the task_id is not found in the workflow.

        Note:
            This is the primary method that should be called to execute a task,
            as it handles all the necessary setup and context gathering.
        """
        print('------Run execute(not _execute)------')
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
        result = await self._execute(task_obj, task_objective, agent_id, context, next_objective)
        ##TODO data, delete this?
        # task_obj.set_history(result)
        return result
