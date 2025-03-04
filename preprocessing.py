import json
from collections import defaultdict
from workflow import Task, Workflow
from history import History


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

    # Populate dependencies and agent-task mappings in a single loop
    for dep in subtask_dependencies:
        dependencies[dep['child']].append(dep['parent'])
    
    for agent in agents:
        agent_id = int(agent['id'].split()[-1])  # Extract agent ID
        for subtask in agent['subtasks']:
            agent_task[subtask].append(agent_id)
    
    # Build workflow and convert to Workflow instance
    tasks = {}
    for i, task in enumerate(subtasks):
        assigned_agent = agent_task[i][0] if agent_task[i] else None
        if assigned_agent is None:
            raise ValueError(f"No agent assigned for task {i}, ensure agent-task mapping is correct.")
        
        task_info = {
            'id': f'task{i}',
            'objective': task['objective'],
            'agent_id': assigned_agent,
            'next': [f'task{key}' for key, value in dependencies.items() if i in value],
            'prev': [f'task{elem}' for elem in dependencies[i]],
            'status': 'pending',
            # 'data': '',
            'history': History(),
            'agent': agents[assigned_agent]['role']
        }
        tasks[f'task{i}'] = Task(**task_info)
    
    return Workflow(tasks)