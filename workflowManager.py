import json
import logging
import concurrent.futures
import logging
from config import Config
import preprocessing
from workflow import Workflow
from gptClient import GPTClient
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
        logger.info("Refining workflow...")
        # TODO the data for updating should include more information, e.g., tasks status and test info
        current_data = {tid: task.__dict__ for tid, task in self.workflow.tasks.items()}
        new_workflow_data = await self.optimize_workflow(current_data)

        # Merge new data and recalculate dependencies
        self.workflow.merge_workflow(new_workflow_data)
        logger.info("Workflow refinement complete.")

    def get_workflow(self) -> Workflow:

        user_content = f'the objective need to be achieved is: {self.objective}'

        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': user_content}
        ]
        response = self.gpt_client.chat_completion(messages)
        logger.info(f"GPT Response: {response}")

        try:
            processed_workflow = preprocessing.process_context(response)
        except Exception as e:
            logger.error(f"IndexError in processing GPT response: {e}")
            raise
        return processed_workflow


    def compare_results(self, all_result):
        if not all_result:
            raise ValueError("No results to compare. All iterations failed.")

        best_workflow = None
        best_score = float('inf')
        dependency_complexities = [workflow.calculate_dependency_complexity() for workflow in all_result]
        parallelisms = [workflow.calculate_average_parallelism() for workflow in all_result]
        print('Comparing...\n Here is the detailed data.')
        print('Dependency complexities: ')
        print(dependency_complexities)
        print('Parallelisms: ')
        print(parallelisms)
            
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

        return best_workflow
    
    def init_workflow(self, n_candidate_graphs=10) -> Workflow:
        """
        Initializes the workflow by generating multiple candidates in parallel,
        then comparing them to select the best candidate.
        """
        results: list[Workflow] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_candidate_graphs) as executor:
            futures = [executor.submit(self.get_workflow) for _ in range(n_candidate_graphs)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    # print(result)
                    if isinstance(result, Workflow):
                        results.append(result)
                except Exception as e:
                    logger.error(f"An error occurred during parallel execution: {e}")

        if not results:
            raise ValueError("All parallel iterations failed. No results to compare.")
        # print(results)
    
        best_workflow = self.compare_results(results)
        self.workflow = best_workflow

        best_workflow.to_json("initflow.json")



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
        response_text = await self.gpt_client.a_chat_completion(messages, temperature=0)
        # Remove any markdown wrappers
        response_text = response_text.strip('```json').strip('```')

        try:
            response_data = json.loads(response_text)
            with open('optimize_log.json', 'a') as file:
                json.dump(response_data, file, indent=4)
                file.write("\n")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nGPT response: {response_text}")
            return current_workflow

        return response_data.get('workflow', current_workflow)
