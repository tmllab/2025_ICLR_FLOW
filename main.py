
import asyncio
import json
import sys
import time
from flow import Flow
import logging
from summary import Summary
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

    overall_task: str = '''I am developing a single player Gobang game in Python language. An graphical user interface and AI player must be implemented.
        AI player will play the game against the human player.
        The game should end when either a player wins or the board is completely filled. 
        The user interface must clearly indicate whose turn it is and display a message when the game concludes, specifying the winner. 
        Additionally, the human player should have the option to play as either black or white stones.
    '''
    refine_threshold: int = 2
    candidate_graphs: int = 3

    
    start_time = time.time()

    manager = Flow(overall_task = overall_task, require_update=False, refine_threhold = refine_threshold, n_candidate_graphs=candidate_graphs)
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
