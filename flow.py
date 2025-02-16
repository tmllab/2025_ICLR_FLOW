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
        redefining (bool): Whether the workflow is currently being redefined.
        task_completion_counter (int): Counts how many tasks have completed since last workflow refinement.
        can_schedule_tasks (asyncio.Event): Controls if scheduling is allowed.
        schedule_lock (asyncio.Lock): Prevents race conditions in scheduling.
        refine_threhold (int): how many tasks have completed to tragger the workflow refinement.
        enable_refine (bool): enable workflow refinment or not
    """

    def __init__(self, overall_task: str, enable_refine=True, refine_threhold=3, n_candidate_graphs=10, workflow = None):
 
        self.overall_task = overall_task
        self.runner = AsyncRunner(overall_task)
        self.optimizer = WorkflowManager(overall_task)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.redefining = False
        self.task_completion_counter = 0
        if workflow:
            self.workflow = workflow
        else:
            self.workflow = self.optimizer.init_workflow(n_candidate_graphs)
        # Controls scheduling of tasks
        self.can_schedule_tasks = asyncio.Event()
        self.can_schedule_tasks.set()

        # Ensures only one scheduling operation at a time
        self.schedule_lock = asyncio.Lock()

        if enable_refine==False:
            self.refine_threhold=float('inf')
        else:
            self.refine_threhold=refine_threhold

    async def run(self):
        """
        Schedule all pending tasks whose dependencies are met.
        """
        if not self.can_schedule_tasks.is_set():
            return

        for task in self.workflow.get_pending_tasks():
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

        if task_obj.status != 'pending':
            logger.info(f"Task {task_id} not pending; skipping.")
            return

        try:
            # Build context from successfully completed (status == 'completed') dependencies


            # Execute task with the runner
            result = await self.runner.execute(self.workflow, task_id)
            task_obj.data = result
            task_obj.status = 'completed'
            self.completed_tasks[task_id] = task_obj

            logger.info(f"Task {task_id} completed with result: {result}")

            # Logging execution result for debugging
            with open('execute_log.json', 'a', encoding='utf-8') as file:
                json.dump({task_id: result}, file, indent=4)

        except asyncio.CancelledError:
            logger.info(f"Task {task_id} was cancelled.")
            raise
        except Exception as e:
            logger.error(f"Task {task_id} encountered error: {e}")
            raise

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
        self.task_completion_counter += 1
        logger.info(
            f"Task {task_id} done. Total completed so far: {self.task_completion_counter}"
        )
        self.workflow.handle_task_completion(task_id)


        # Trigger workflow refinement when threshold is reached
        if self.task_completion_counter >= self.refine_threhold and not self.redefining:
            logger.info(f"Task {task_id} triggers workflow refinement.")
            self.task_completion_counter = 0
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
            logger.info(f" - {task_id}: {task_obj.data}")




