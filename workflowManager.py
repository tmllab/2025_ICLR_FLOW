import json
import concurrent.futures
from config import Config
import preprocessing
from workflow import Workflow
from gptClient import GPTClient
import prompt
from logging_config import get_logger, log_workflow_event, log_intermediate_result, save_intermediate_snapshot, get_run_directory


# -----------------------------------------------------------------------------
# Configuration and Logging Setup
# -----------------------------------------------------------------------------
logger = get_logger('workflow')


# -----------------------------------------------------------------------------
# Workflow Optimizer
# -----------------------------------------------------------------------------
class WorkflowManager:
    """Uses GPT to optimize and update the workflow."""

    def __init__(self, objective: str):
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.objective = objective
        self.workflow: Workflow | None = None
        self.system_prompt = prompt.INIT_WORKFLOW_PROMPT



    async def update_workflow(self) -> None:
        """
        Refines the current workflow based on completed tasks.
        """
        logger.info("Starting workflow refinement...")
        
        # Log pre-refinement state as intermediate result (properly serialize History objects)
        current_data = {}
        for tid, task in self.workflow.tasks.items():
            task_dict = task.__dict__.copy()
            # Convert History object to dict for JSON serialization
            if hasattr(task_dict['history'], 'to_dict'):
                task_dict['history'] = task_dict['history'].to_dict()
            current_data[tid] = task_dict
        
        log_intermediate_result(
            task_id="workflow_refinement",
            iteration=1,
            result_type="workflow_pre_refinement",
            data={
                "total_tasks": len(current_data),
                "completed_tasks": sum(1 for task in current_data.values() if task.get('status') == 'completed'),
                "pending_tasks": sum(1 for task in current_data.values() if task.get('status') == 'pending'),
                "failed_tasks": sum(1 for task in current_data.values() if task.get('status') == 'failed')
            },
            status="refinement_starting"
        )
        
        # Save pre-refinement workflow snapshot
        save_intermediate_snapshot(
            "workflow_pre_refinement.json",
            {
                "workflow_tasks": current_data
            },
            "Workflow state before refinement process",
            task_id="workflow_refinement",
            iteration=1
        )
        
        new_workflow_response = await self.optimize_workflow(current_data)

        # Extract the workflow tasks from the GPT response  
        if isinstance(new_workflow_response, dict):
            if 'workflow' in new_workflow_response:
                new_workflow_data = new_workflow_response['workflow']
            else:
                # Response is already the tasks dict
                new_workflow_data = new_workflow_response
        else:
            logger.error(f"Unexpected response type from optimize_workflow: {type(new_workflow_response)}")
            logger.error(f"Response content: {new_workflow_response}")
            return
            
        # Validate that new_workflow_data is a dictionary
        if not isinstance(new_workflow_data, dict):
            logger.error(f"Expected workflow data to be dict, got {type(new_workflow_data)}: {new_workflow_data}")
            return

        # Merge new data and recalculate dependencies
        self.workflow.merge_workflow(new_workflow_data)
        
        # Log post-refinement state as intermediate result (properly serialize History objects)
        updated_data = {}
        for tid, task in self.workflow.tasks.items():
            task_dict = task.__dict__.copy()
            # Convert History object to dict for JSON serialization
            if hasattr(task_dict['history'], 'to_dict'):
                task_dict['history'] = task_dict['history'].to_dict()
            updated_data[tid] = task_dict
        
        log_intermediate_result(
            task_id="workflow_refinement",
            iteration=2,
            result_type="workflow_post_refinement",
            data={
                "total_tasks_after": len(updated_data),
                "new_tasks_added": len(updated_data) - len(current_data),
                "refinement_completed": True
            },
            status="refinement_completed"
        )
        
        # Save post-refinement workflow snapshot
        save_intermediate_snapshot(
            "workflow_post_refinement.json",
            {
                "refined_workflow_tasks": updated_data,
                "changes_summary": {
                    "tasks_before": len(current_data),
                    "tasks_after": len(updated_data),
                    "net_change": len(updated_data) - len(current_data)
                }
            },
            "Workflow state after refinement process",
            task_id="workflow_refinement",
            iteration=2
        )
        
        logger.info("Workflow refinement complete.")

    def get_workflow(self) -> Workflow:

        user_content = f'the objective need to be achieved is: {self.objective}'

        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': user_content}
        ]
        response_format = {"type": "json_object"}
        response = self.gpt_client.chat_completion(messages, response_format=response_format)
        logger.info(f"GPT Response: {response}")

        try:
            processed_workflow = preprocessing.process_context(response)
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Error processing GPT response: {e}")
            logger.error(f"Raw GPT response: {response}")
            # Provide more context about what went wrong
            error_msg = f"Failed to process workflow from GPT response: {str(e)}\n"
            error_msg += f"This usually means GPT returned unexpected JSON structure.\n"
            error_msg += f"Expected keys: ['subtasks', 'subtask_dependencies', 'agents']\n"
            error_msg += f"Raw response: {response[:500]}..." if len(response) > 500 else f"Raw response: {response}"
            raise ValueError(error_msg) from e
        except Exception as e:
            logger.error(f"Unexpected error in processing GPT response: {e}")
            logger.error(f"Raw GPT response: {response}")
            raise
        return processed_workflow


    def compare_results(self, all_result):
        if not all_result:
            raise ValueError("No results to compare. All iterations failed.")

        best_workflow = None
        best_score = float('inf')
        dependency_complexities = [workflow.calculate_dependency_complexity() for workflow in all_result]
        parallelisms = [workflow.calculate_average_parallelism() for workflow in all_result]
        logger.info(f'Comparing {len(all_result)} workflow candidates...')
        logger.info(f'Dependency complexities: {dependency_complexities}')
        logger.info(f'Parallelisms: {parallelisms}')
            
        mean_dependency_complexity = sum(dependency_complexities) / len(dependency_complexities)
        mean_parallelism = sum(parallelisms) / len(parallelisms)
        
        std_dependency_complexity = (sum((x - mean_dependency_complexity) ** 2 for x in dependency_complexities) / len(dependency_complexities)) ** 0.5
        std_parallelism = (sum((x - mean_parallelism) ** 2 for x in parallelisms) / len(parallelisms)) ** 0.5
        
        epsilon = 0.01
        
        for workflow in all_result:
            
            z_dependency_complexity = (workflow.calculate_dependency_complexity() - mean_dependency_complexity) / (std_dependency_complexity + epsilon)
            z_parallelism = (workflow.calculate_average_parallelism() - mean_parallelism) / (std_parallelism + epsilon)
            
            score = z_dependency_complexity - z_parallelism
            if score < best_score:
                best_score = score
                best_workflow = workflow

        # Save workflow candidates comparison as intermediate result
        log_intermediate_result(
            task_id="workflow_initialization",
            iteration=1,
            result_type="workflow_candidates_comparison",
            data={
                "total_candidates": len(all_result),
                "dependency_complexities": dependency_complexities,
                "parallelisms": parallelisms,
                "best_workflow_score": best_score,
                "mean_dependency_complexity": mean_dependency_complexity,
                "mean_parallelism": mean_parallelism
            },
            status="candidate_selection_completed"
        )

        return best_workflow
    
    def init_workflow(self, n_candidate_graphs=10) -> Workflow:
        """
        Initializes the workflow by generating multiple candidates in parallel,
        then comparing them to select the best candidate.
        """
        results: list[Workflow] = []

        failed_attempts = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_candidate_graphs) as executor:
            futures = [executor.submit(self.get_workflow) for _ in range(n_candidate_graphs)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if isinstance(result, Workflow):
                        results.append(result)
                        logger.info(f"Successfully generated workflow candidate {len(results)}")
                except Exception as e:
                    failed_attempts += 1
                    logger.error(f"Failed to generate workflow candidate {failed_attempts}: {str(e)[:200]}...")
                    
                    # Log more detail for first few failures
                    if failed_attempts <= 3:
                        logger.error(f"Detailed error for attempt {failed_attempts}: {e}")

        logger.info(f"Workflow generation complete: {len(results)} successful, {failed_attempts} failed")
        
        if not results:
            error_msg = f"All {n_candidate_graphs} parallel workflow generation attempts failed. "
            error_msg += "This usually indicates an issue with GPT response format or JSON parsing. "
            error_msg += "Check the logs above for specific error details."
            raise ValueError(error_msg)
    
        best_workflow = self.compare_results(results)
        self.workflow = best_workflow

        # Save initflow.json to the run directory
        initflow_path = get_run_directory() / "initflow.json"
        best_workflow.to_json(str(initflow_path))
        logger.info(f"Initial workflow saved to {initflow_path}")



        return best_workflow

    async def optimize_workflow(self, current_workflow: dict) -> dict:
        """
        Evaluates and optimizes the current workflow using GPT.
        """
        # Simplify workflow to only necessary attributes
        simplified_workflow = {
            task_id: {
                'objective': task_info.get('objective', {}),
                'agent_id': task_info.get('agent_id', -1),
                'next': task_info.get('next', []),
                'prev': task_info.get('prev', []),
                'status': task_info.get('status', ''),
                # TODO: justify here
                # 'data': task_info.get('data', '')
            }
            for task_id, task_info in current_workflow.items()
        }
        simplified_workflow['final_goal'] = self.objective.strip()

        logger.info("Sending request to GPT for optimization...")
        
        messages = [
            {"role": "system", "content": prompt.UPDATE_WORKFLOW_PROMPT},
            {"role": "user", "content": json.dumps(simplified_workflow, indent=4)}
        ]
        response_format = {"type": "json_object"}
        response_text = await self.gpt_client.a_chat_completion(messages, response_format=response_format)

        try:
            response_data = json.loads(response_text)
            log_workflow_event(
                event_type='workflow_optimization',
                data={
                    'response_data': response_data,
                    'simplified_workflow_size': len(simplified_workflow),
                    'optimization_requested': True
                }
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nGPT response: {response_text}")
            return current_workflow

        # Ensure we return the workflow tasks dictionary
        workflow_data = response_data.get('workflow', current_workflow)
        if workflow_data is None:
            logger.warning("GPT response contained no workflow data, using current workflow")
            return current_workflow
            
        return workflow_data
