
import asyncio
import json
import sys
import time
from flow import Flow
import logging
from summary import Summary
from task_prompt import *
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
# Main Function
# -----------------------------------------------------------------------------
def main():
    """
    Entry point for running the workflow. Defines the overall task, creates an initial workflow,
    and orchestrates the manager.
    """
    # Ensure UTF-8 encoding for stdout (optional, depending on environment)
    sys.stdout.reconfigure(encoding='utf-8')


    

    overall_task: str = tetris_bjeweled

    candidate_graphs: int = 5
    refine_threshold: int = 3
    max_refine_itt=5
    max_validation_itt: int = 10


    # Record the whole validation process in a new overall task, following the previous one
    with open('validate_log.json', 'a', encoding='utf-8') as file:
            file.write(f'\n**********\nHere is the whole validation process of a new overall task:\n{overall_task}\n**********\n')

    start_time = time.time()

    manager = Flow(overall_task=overall_task, refine_threshold=refine_threshold,max_refine_itt=max_refine_itt,n_candidate_graphs=candidate_graphs,workflow=None,max_validation_itt=max_validation_itt)
    asyncio.run(manager.run_async())

    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")


    workflow_data = manager.workflow.to_dict()
  
    # with open('result.json', 'w', encoding='utf-8') as file:
    #     json.dump(workflow_data, file, indent=4)

    summary = Summary()
    # Generate and save a summary of the workflow results
    chat_result = summary.summary(overall_task, workflow_data)
    with open("example.txt", "w", encoding="utf-8") as file:
        file.write(chat_result)


if __name__ == "__main__":
    main()
