
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

    # overall_task: str = '''I want to create a website for the following conference:
    #     1). Conference Name: International Conference on Learning Representations (ICLR2025)  
    #     2). Date: April 27, 2025 to May 1, 2025  
    #     3). Location: San Francisco, California, United States
    #     4). Organizer: International Association for Learning Representations
    #     Please generate a detailed website structure and content for this conference. 
    #     For each section, provide example HTML content. 
    #     Additionally, create a sample CSS stylesheet to style the website. 
    #     Ensure the content is professional, clear, and suitable for an international academic conference.
    #     Note that:
    #     1). The previous information I gave you must be included.
    #     2). The website should have conference schedule part.
    #     3). The website should have conference venue part with a map.
    #     '''
    
    # overall_task = '''1. Lecture slide:
    # I am a lecturer. I am teaching the machine learning coure for research students. Please generate latex code for lecture slide for different reinforcement learning algorithms.
    # Note that:
    # 1). Note that the lecture duration is 2 hour, so we need to generate 30 pages.
    # 2). for each reinforcement learning algorithms, the slide should include motivation, problem and intuitive solution and detailed math equations.
    # 3). Please make sure the the lecture have a good self-contain.
    # '''
    overall_task: str = '''Develop a game that fuses Tetris and Bejeweled mechanics. 
    Falling tetrominoes should lock into a grid and transform into colored gems. 
    The game must support both Tetris line-clearing and Bejeweled match-3 clearing, triggering chain reactions and bonus points. 
    Include a GUI (using a framework like Pygame) that displays the game grid, current score, and next tetromino preview, along with smooth animations. 
    No sound effects are needed.'''
    # overall_task: str = '''Develop a Snake game with a graphical user interface (GUI) in Python.'''
    # overall_task: str = '''Develop a Rock-Paper-Scissors game with a graphical user interface (GUI) in Python. 
    # The game should allow a player to compete against a naive AI that randomly selects Rock, Paper, or Scissors. 
    # The UI should display the player's choice, the AI's choice, and the game result (win, lose, or draw). 
    # Provide an interactive and user-friendly experience.'''
    
    # Record the whole validation process in a new overall task, following the previous one
    with open('validate_log.json', 'a', encoding='utf-8') as file:
            file.write(f'\n**********\nHere is the whole validation process of a new overall task:\n{overall_task}\n**********\n')

    start_time = time.time()

    manager = Flow(overall_task = overall_task, refine_threhold = 3, max_refine_itt = 5, n_candidate_graphs=5,workflow=None,max_validation_itt=10)
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
