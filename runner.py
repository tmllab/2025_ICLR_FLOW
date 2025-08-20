from taskexecuter import taskExecuter
import logging
from config import Config
from workflow import Workflow
from autovalidator import Validator
from workflow import Task
import prompt
from logging_config import get_logger, log_intermediate_result, save_intermediate_snapshot
import time
 
# -----------------------------------------------------------------------------
# Configuration and Logging Setup
# -----------------------------------------------------------------------------
logger = get_logger('execution')

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
        
        start_time = time.time()
        result = await self.executer.execute(task_objective, agent_id, context, next_objective, task_obj.output_format)
        execution_time = time.time() - start_time
        
        # Log initial execution result as intermediate result
        log_intermediate_result(
            task_id=task_id,
            iteration=0,
            result_type="task_execution",
            data={
                "task_objective": task_objective,
                "agent_id": agent_id,
                "execution_time": execution_time,
                "result_length": len(result),
                "result_preview": result[:200] + "..." if len(result) > 200 else result
            },
            status="executed"
        )
        
        # Save initial execution snapshot
        save_intermediate_snapshot(
            f"task_execution_{task_id}.json",
            {
                "task_objective": task_objective,
                "agent_id": agent_id,
                "context": context,
                "next_objective": next_objective,
                "result": result,
                "execution_time": execution_time
            },
            f"Initial execution result for task {task_id}",
            task_id=task_id,
            iteration=0
        )
        
        # If validation is disabled, mark as completed immediately
        if self.max_validation_itt == 0:
            task_obj.save_history(result, '')
            task_obj.set_status("completed")
            return result

        # Validation loop with re-execution on failure
        for iteration in range(self.max_validation_itt):
            logger.info(f"Validation iteration {iteration + 1}/{self.max_validation_itt} for task '{task_id}'")
            
            # Validate the current result (now includes overall task context and output format)
            validation_start = time.time()
            feedback, new_status = await self.validator.validate(task_obj.objective, result, task_obj.get_history(), self.overall_task, task_obj.output_format)
            validation_time = time.time() - validation_start
            
            # Log validation attempt as intermediate result
            log_intermediate_result(
                task_id=task_id,
                iteration=iteration + 1,
                result_type="validation_attempt",
                data={
                    "validation_time": validation_time,
                    "status": new_status,
                    "feedback_length": len(feedback),
                    "feedback_preview": feedback[:200] + "..." if len(feedback) > 200 else feedback
                },
                status=new_status
            )
            
            # Save validation snapshot
            save_intermediate_snapshot(
                f"validation_{task_id}.json",
                {
                    "validation_iteration": iteration + 1,
                    "result_being_validated": result,
                    "validation_status": new_status,
                    "validation_feedback": feedback,
                    "validation_time": validation_time,
                    "task_history": task_obj.get_history()
                },
                f"Validation attempt {iteration + 1} for task {task_id}",
                task_id=task_id,
                iteration=iteration + 1
            )
            
            if new_status == 'completed':
                logger.info(f"Task '{task_id}' validated successfully after {iteration + 1} iteration(s)")
                task_obj.save_history(result, feedback)
                task_obj.set_status("completed")
                
                # Log successful completion
                log_intermediate_result(
                    task_id=task_id,
                    iteration=iteration + 1,
                    result_type="task_completion",
                    data={
                        "total_attempts": iteration + 1,
                        "final_status": "completed",
                        "final_result_length": len(result)
                    },
                    status="completed"
                )
                
                return result
            
            # Validation failed - save the failed attempt
            task_obj.save_history(result, feedback)
            logger.info(f"Validation failed for task '{task_id}': {feedback[:100]}...")
            
            # Re-execute for the next validation attempt (except on last iteration)
            if iteration < self.max_validation_itt - 1:
                logger.info(f"Re-executing task '{task_id}' for attempt {iteration + 2}")
                
                re_exec_start = time.time()
                result = await self.executer.re_execute(task_objective, context, next_objective, result, task_obj.get_history(), task_obj.output_format)
                re_exec_time = time.time() - re_exec_start
                
                # Log re-execution as intermediate result
                log_intermediate_result(
                    task_id=task_id,
                    iteration=iteration + 2,
                    result_type="task_re_execution",
                    data={
                        "re_execution_time": re_exec_time,
                        "attempt_number": iteration + 2,
                        "result_length": len(result),
                        "result_preview": result[:200] + "..." if len(result) > 200 else result
                    },
                    status="re_executed"
                )
                
                # Save re-execution snapshot
                save_intermediate_snapshot(
                    f"re_execution_{task_id}.json",
                    {
                        "re_execution_attempt": iteration + 2,
                        "previous_result": task_obj.get_latest_history()[0] if task_obj.get_history() else "",
                        "previous_feedback": task_obj.get_latest_history()[1] if task_obj.get_history() else "",
                        "new_result": result,
                        "re_execution_time": re_exec_time,
                        "task_history": task_obj.get_history()
                    },
                    f"Re-execution attempt {iteration + 2} for task {task_id}",
                    task_id=task_id,
                    iteration=iteration + 2
                )

        # If we exit the loop, all validation attempts failed
        logger.warning(f"Task '{task_id}' failed validation after {self.max_validation_itt} attempts. Marking as failed.")
        task_obj.set_status("failed")
        
        # Log final failure state
        log_intermediate_result(
            task_id=task_id,
            iteration=self.max_validation_itt,
            result_type="task_failure",
            data={
                "total_attempts": self.max_validation_itt,
                "final_status": "failed",
                "final_result_length": len(result)
            },
            status="failed"
        )
        
        return result
