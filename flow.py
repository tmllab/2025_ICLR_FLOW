import asyncio
import json
import logging
from typing import Dict, Any
from runner import AsyncRunner
from workflowManager import WorkflowManager

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
# Workflow Manager
# -----------------------------------------------------------------------------
class Flow:
    """
    Orchestrates the entire multi-agent workflow:
      - Schedules tasks whose dependencies are met.
      - Monitors task execution.
      - Triggers workflow refinements based on a threshold.

    Attributes:
        workflow (Workflow): The workflow object containing tasks.
        overall_task (str): Description of the overall task.
        runner (AsyncRunner): An asynchronous runner that executes tasks.
        optimizer (WorkflowManager): An instance used to refine the workflow.
        active_tasks (Dict[str, asyncio.Task]): Maps task IDs to active asyncio tasks.
        completed_tasks (Dict[str, asyncio.Task]): Maps task IDs to completed tasks.
        redefining (bool): Whether the workflow is currently
        task_done_counter (int): Counts how many tasks have completed since last workflow refinement.
        can_schedule_tasks (asyncio.Event): Controls if scheduling is allowed.
        schedule_lock (asyncio.Lock): Prevents race conditions in scheduling.
        refine_threshold (int): how many tasks have completed to tragger the workflow refinement.
        max_validation_itt(int): how many times the validation work will repeat.
    """

    def __init__(self, overall_task: str, refine_threshold=3, max_refine_itt = 5, n_candidate_graphs=10, workflow = None, max_validation_itt: int = 0):
 
        self.overall_task = overall_task
        self.runner = AsyncRunner(overall_task, max_validation_itt)
        self.optimizer = WorkflowManager(overall_task)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.redefining = False
        self.task_done_counter = 0
        self.max_refine_itt = max_refine_itt
        if workflow is not None:
            self.workflow = workflow
        else:
            self.workflow = self.optimizer.init_workflow(n_candidate_graphs)
        # Controls scheduling of tasks
        self.can_schedule_tasks = asyncio.Event()
        self.can_schedule_tasks.set()

        # Ensures only one scheduling operation at a time
        self.schedule_lock = asyncio.Lock()
        self.refine_threshold=refine_threshold

    async def run(self):
        """
        Schedule all pending tasks whose dependencies are met.
        """
        if not self.can_schedule_tasks.is_set():
            return

        for task in self.workflow.get_runable_tasks():     
            await self.schedule_task(task.id)
      
    async def schedule_task(self, task_id: str):
        """
        Schedule a single task for execution if it's not already active.
        """
        async with self.schedule_lock:
            if not self.can_schedule_tasks.is_set():
                return
            if task_id in self.active_tasks:
                return

            task_coroutine = self.run_task(task_id)
            task = asyncio.create_task(task_coroutine)
            self.active_tasks[task_id] = task
            # Start monitoring this task
            asyncio.create_task(self.monitor_task(task_id, task))

    async def monitor_task(self, task_id: str, task: asyncio.Task):
        """
        Monitor a running task and handle its completion or error.
        """
        try:
            await task
        except asyncio.CancelledError:
            self.task_cancelled_callback(task_id)
        except Exception as e:
            self.task_error_callback(task_id, e)
        else:
            await self.task_done_callback(task_id)

    async def run_task(self, task_id: str):
        """
        Core logic to run an individual task in the workflow.

        1. Skip if not pending.
        2. Build context from completed dependency tasks.
        3. Execute the task using AsyncRunner.
        4. Save and log the result.
        """
        task_obj = self.workflow.tasks[task_id]

        if task_obj.status == 'completed':
            logger.info(f"Task {task_id} completed; skipping.")
            return

      
        # Execute task with the runner and set status = 'completed' or 'failed'
        result = await self.runner.execute(self.workflow, task_id)


        # Logging execution result for debugging
        with open('execute_log.json', 'a', encoding='utf-8') as file:
            json.dump({task_id: result}, file, indent=4)


    def task_cancelled_callback(self, task_id: str):
        """
        Cleanup logic for a cancelled task.
        """
        self.active_tasks.pop(task_id, None)
        logger.info(f"Cancelled task {task_id} cleaned up.")

    def task_error_callback(self, task_id: str, error: Exception):
        """
        Cleanup logic for a task that errored out.
        """
        self.active_tasks.pop(task_id, None)
        logger.error(f"Task {task_id} encountered an error: {error}")

    async def task_done_callback(self, task_id: str):
        """
        Called when a task completes successfully. 
        Increments counters, updates downstream tasks, and triggers workflow refinement if needed.
        """
        self.active_tasks.pop(task_id, None)
        self.task_done_counter += 1
        logger.info(
            f"Task {task_id} done. Total completed so far: {self.task_done_counter}"
        )
        self.workflow.handle_task_done(task_id)


        # Trigger workflow refinement when threshold is reached
        print(self.task_done_counter , self.refine_threshold )
        if self.task_done_counter >= self.refine_threshold and not self.redefining and self.max_refine_itt > 0:
            logger.info(f"Task {task_id} triggers workflow refinement.")
            self.task_done_counter = 0
            self.redefining = True
            self.can_schedule_tasks.clear()

            # Wait for all currently active tasks to finish before refining
            if self.active_tasks:
                logger.info("Waiting for active tasks to complete before refinement.")
                await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

            # Redefine workflow
            await self.optimizer.update_workflow()
            self.can_schedule_tasks.set()
            self.redefining = False
            self.max_refine_itt -= 1

        # Continue scheduling any remaining tasks
        await self.run()


    async def run_async(self):
        """
        Continuously runs tasks until the entire workflow is complete.
        """
        while not self.workflow.all_completed():
            await self.run()
            await asyncio.sleep(0.1)

        logger.info("All tasks completed. Final Task Results:")
        for task_id, task_obj in sorted(self.workflow.tasks.items()):
            result, feedback = task_obj.get_latest_history()
            logger.info(f" - {task_id}: {result}")




