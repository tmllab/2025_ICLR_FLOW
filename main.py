
import asyncio
import json
import sys
import time
from flow import Flow
import logging
from summary import Summary
from config import global_context
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

    overall_task: str = '''I am designing a website for the International Conference on Learning Representations (ICLR2025), which will take place from April 27, 2025, to May 1, 2025, in San Francisco, California, United States. The conference is organized by the International Association for Learning Representations.
                            Note that:
                            1). For each section, I would like to see example HTML content. Additionally, a sample CSS stylesheet should be provided to style the website. The content must be professional, clear, and appropriate for an international academic conference.
                            2). The website should include all the provided details, including a comprehensive conference schedule and a section dedicated to the conference venue, featuring a map.
                        '''



    start_time = time.time()

<<<<<<< Updated upstream
    manager = Flow(overall_task = overall_task, enable_refine=False, refine_threhold = 3, n_candidate_graphs=10,workflow=None)
=======
    manager = Flow(overall_task = overall_task, enable_refine=False, refine_threhold = 3, n_candidate_graphs=3,workflow=None,max_itt=3)
>>>>>>> Stashed changes
    asyncio.run(manager.run_async())

    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")


    workflow_data = {
        tid: task.__dict__ for tid, task in manager.workflow.tasks.items()
    }
    with open('result.json', 'w', encoding='utf-8') as file:
        json.dump(workflow_data, file, indent=4)

    summary = Summary()
    # Generate and save a summary of the workflow results
    chat_result = summary.summary(overall_task, workflow_data)
    with open("example.txt", "w", encoding="utf-8") as file:
        file.write(chat_result)


if __name__ == "__main__":
    main()
