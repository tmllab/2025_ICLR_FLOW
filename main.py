
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

    # Define overall workflow
    overall_task: str = '''Develop a Rock-Paper-Scissors game with a graphical user interface (GUI) in Python. 
    The game should allow a player to compete against a naive AI that randomly selects Rock, Paper, or Scissors. 
    The UI should display the player's choice, the AI's choice, and the game result (win, lose, or draw). 
    Provide an interactive and user-friendly experience.'''

    # Set optimization threshold and number of candidate graphs
    refine_threshold: int = 3
    candidate_graphs: int = 10

    # Start the timer
    start_time = time.time()

    # Run FLOW
    manager = Flow(overall_task = overall_task, enable_refine=False, refine_threhold = refine_threshold, n_candidate_graphs=candidate_graphs,workflow=None)
    asyncio.run(manager.run_async())

    # Stop the timer
    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Store workflow
    workflow_data = {
        tid: task.__dict__ for tid, task in manager.workflow.tasks.items()
    }
    with open('result.json', 'w', encoding='utf-8') as file:
        json.dump(workflow_data, file, indent=4)

    # Generate and save a summary of the workflow results
    summary = Summary()
    chat_result = summary.summary(overall_task, workflow_data)
    with open("example.txt", "w", encoding="utf-8") as file:
        file.write(chat_result)


if __name__ == "__main__":
    main()
