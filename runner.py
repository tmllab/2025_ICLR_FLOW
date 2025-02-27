from taskexecuter import taskExecuter
import logging
from config import Config
from workflow import Workflow
from validator import Validator
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
    """Executes an individual task asynchronously via GPT."""
    def __init__(self, overall_task: str, max_itt):
        self.overall_task = overall_task
        self.max_itt = max_itt
        self.executer = taskExecuter(overall_task)
        self.validator = Validator()

    async def _execute(self, subtask: str, agent_id: str, context: str, next_objective: str) -> str:
        print('------Run _execute------')
        i = 0
        status = "pending"
        # execute here

        while i < self.max_itt:
            print(f'------Go into while loop, i: {i}------')
            if i == 0:
                result = await self.executer.execute(subtask, agent_id, context, next_objective)
            else:
                # re-execute here
                # TODO: paramaters are wrong
                result = await self.executer.re_execute(subtask, agent_id, context, next_objective, feedback)

            feedback, new_status = await self.validator.validate(subtask, result)
            ##TODO data
            ##TODO status 
            status = new_status
            if status == "complete" :
                print('---Result is perfect---')
              
                break

 
            
            i += 1
        ##TODO update task status here
        #  
        return result

    async def execute(self, workflow: Workflow, task_id: str) -> str:
        print('------Run execute(not _execute)------')
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
