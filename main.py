
import asyncio
import json
import sys
import time
from flow import Flow
from summary import Summary
from task_prompt import *
from logging_config import get_logger, log_workflow_summary, save_results, save_run_metadata, copy_workflow_file, get_run_directory, get_run_id

# -----------------------------------------------------------------------------
# Configuration and Logging Setup
# -----------------------------------------------------------------------------
logger = get_logger('main')



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


    

    overall_task: str = '''Develop a game that fuses Tetris and Bejeweled mechanics. 
This game needs to add keyboard control function.
Falling tetrominoes should lock into a grid and transform into colored gems. 
The game must support both Tetris line-clearing and Bejeweled match-3 clearing, triggering chain reactions and bonus points. 
Include a GUI (using a framework like Pygame) that displays the game grid, current score, and next tetromino preview, along with smooth animations. 
No sound effects are needed.

Output Format: python'''

    candidate_graphs: int = 5
    refine_threshold: int = 3
    max_refine_itt=5
    max_validation_itt: int = 5


    # Save run metadata and configuration
    config = {
        "candidate_graphs": candidate_graphs,
        "refine_threshold": refine_threshold,
        "max_refine_itt": max_refine_itt,
        "max_validation_itt": max_validation_itt
    }
    save_run_metadata(overall_task, config)
    
    logger.info(f"Starting Flow execution for task: {overall_task[:100]}...")
    logger.info(f"Run ID: {get_run_id()}")
    logger.info(f"Run Directory: {get_run_directory()}")
    logger.info(f"Configuration: {candidate_graphs} candidates, refine_threshold={refine_threshold}, max_validation_itt={max_validation_itt}")

    start_time = time.time()

    manager = Flow(overall_task=overall_task, refine_threshold=refine_threshold,max_refine_itt=max_refine_itt,n_candidate_graphs=candidate_graphs,workflow=None,max_validation_itt=max_validation_itt)
    asyncio.run(manager.run_async())

    elapsed_time = time.time() - start_time
    logger.info(f"Flow execution completed in {elapsed_time:.2f} seconds")

    workflow_data = manager.workflow.to_dict()
    
    # Save workflow data with improved naming
    save_results(
        "workflow_final_state.json", 
        workflow_data, 
        f"Final workflow state for task: {overall_task[:50]}..."
    )

    summary = Summary()
    # Generate and save a summary of the workflow results
    chat_result = summary.summary(overall_task, workflow_data)
    
    # Save summary with better naming and metadata
    save_results(
        "final_summary.json",
        {"summary_text": chat_result, "original_task": overall_task},
        "Final synthesized summary of workflow execution"
    )
    
    # Also save as text file for easy reading in the run directory
    example_file = get_run_directory() / "final_summary.txt"
    with open(example_file, "w", encoding="utf-8") as file:
        file.write(chat_result)
    
    # Log complete workflow summary
    log_workflow_summary(overall_task, workflow_data, elapsed_time, chat_result)
    
    logger.info(f"All results saved successfully to run directory: {get_run_directory()}")
    logger.info(f"Run completed! Check {get_run_directory()} for all files from this execution.")


if __name__ == "__main__":
    main()
