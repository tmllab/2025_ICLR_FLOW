import json
from collections import defaultdict
from workflow import Workflow
from task import Task

# read json
def readjson(context: dict):
    task = context['task']
    subtasks = context['subtasks']
    subtask_dependencies = context['subtask_dependencies']
    agents = context['agents']
    return subtasks, subtask_dependencies, agents



def process_context(context: str):
    # Strip and parse JSON
    context = json.loads(context.strip('```json').strip('```'))

    # Extract relevant data
    subtasks = context['subtasks']
    subtask_dependencies = context['subtask_dependencies']
    agents = context['agents']

    # Initialize dependencies and agent-task mappings
    dependencies = defaultdict(list)
    agent_task = defaultdict(list)

    # Populate dependencies and agent-task mappings
    for dep in subtask_dependencies:
        dependencies[dep['child']].append(dep['parent'])

    for agent in agents:
        agent_id = int(agent['id'].split()[-1])  # Extract agent ID
        for subtask in agent['subtasks']:
            agent_task[subtask].append(agent_id)

    # Step 1: Create all tasks without next and prev
    tasks = {}
    for i, task in enumerate(subtasks):
        assigned_agent = agent_task[i][0] if agent_task[i] else None
        if assigned_agent is None:
            raise ValueError(f"No agent assigned for task {i}, ensure agent-task mapping is correct.")

        tasks[f'task{i}'] = Task(
            id=f'task{i}',
            objective=task['objective'],
            agent_id=assigned_agent,
            agent=agents[assigned_agent]['role']
        )

    # Step 2: Connect tasks by replacing next and prev with actual task objects
    for i, task in enumerate(subtasks):
        task_obj = tasks[f'task{i}']
        task_obj.next = [tasks[f'task{key}'] for key, value in dependencies.items() if i in value]
        task_obj.prev = [tasks[f'task{elem}'] for elem in dependencies[i]]

    return Workflow(tasks)