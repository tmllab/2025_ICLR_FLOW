from workflow import Task, Workflow

# Customize your overall task here.
overall_task: str = '''Develop a Rock-Paper-Scissors game with a graphical user interface (GUI) in Python. The game should allow a player to compete against a naive AI that randomly selects Rock, Paper, or Scissors. The UI should display the player’s choice, the AI’s choice, and the game result (win, lose, or draw). Provide an interactive and user-friendly experience.'''

# Use a list of dictionaries to initialize task data here.
tasks_data = [
    {
            "id": "task0",
            "objective": "Design the overall architecture of the Rock-Paper-Scissors game in python. This includes defining the components such as the user interface, game logic, and AI decision-making process. The description should provide clarity on how each component interacts within the system.",
            "agent_id": 0,
            "next": [
                "task1",
                "task2",
                "task3"
            ],
            "prev": [],
            "agent": "Game Architect"
        },
    {
            "id": "task1",
            "objective": "Implement the graphical user interface (GUI) for the Rock-Paper-Scissors game in python. The GUI should allow users to select their choice, display the AI's choice, and show the game results. Provide a detailed description of the UI elements and user experience design.",
            "agent_id": 1,
            "next": [
                "task4"
            ],
            "prev": [
                "task0"
            ],
            "agent": "UI Developer"
        },
    {
            "id": "task2",
            "objective": "Develop the game logic that determines the winner based on the player's choice and the AI's choice in python. This task requires a clear explanation of the rules of the game, how outcomes are calculated, and how results are communicated to the GUI.",
            "agent_id": 2,
            "next": [
                "task4"
            ],
            "prev": [
                "task0"
            ],
            "agent": "Game Logic Developer"
        },
    {
            "id": "task3",
            "objective": "Create the naive AI that randomly selects Rock, Paper, or Scissors in python. This subtask involves defining the algorithm for random selection and ensuring that it integrates smoothly with the game logic and GUI.",
            "agent_id": 3,
            "next": [
                "task4"
            ],
            "prev": [
                "task0"
            ],
            "agent": "AI Developer"
        },
    {
            "id": "task4",
            "objective": "Integrate the GUI with the game logic and AI components in python. This entails ensuring that user interactions are processed correctly, the AI's selections are displayed, and the game results are updated in real-time within the GUI.",
            "agent_id": 4,
            "next": [
                "task5"
            ],
            "prev": [
                "task1",
                "task2",
                "task3"
            ],
            "agent": "Integration Specialist"
        }
]

# Create the workflow
workflow = Workflow(tasks={task_data["id"]: Task(**task_data) for task_data in tasks_data})


import asyncio
import json
from flow import Flow
from summary import Summary


manager = Flow(overall_task=overall_task,  refine_threhold=0, n_candidate_graphs=0, workflow=workflow)
asyncio.run(manager.run_async())


workflow_data = {
    tid: task.__dict__ for tid, task in manager.workflow.tasks.items()
}
with open('manually_result.json', 'w', encoding='utf-8') as file:
    json.dump(workflow_data, file, indent=4)

summary = Summary()

# Generate and save a summary of the workflow results
chat_result = summary.summary(overall_task, workflow_data)
with open("example.txt", "w", encoding="utf-8") as file:
    file.write(chat_result)