import json
from collections import defaultdict
from workflow import Task, Workflow


# read json
def readjson(context: dict):
    task = context['task']
    subtasks = context['subtasks']
    subtask_dependencies = context['subtask_dependencies']
    agents = context['agents']
    return subtasks, subtask_dependencies, agents

def process_context(context: str):

    context = context.strip('```json').strip('```')
    
    try:
        context = json.loads(context)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse GPT response as JSON.")
    
    subtasks = context['subtasks']
    subtask_dependencies = context['subtask_dependencies']
    agents = context['agents']
    
    # Initialize the dictionaries for dependencies and agent-task mapping
    dependencies = defaultdict(list)
    agent_task = defaultdict(list)

    # Populate dependencies and agent-task mappings
    for dependency in subtask_dependencies:
        dependencies[dependency['child']].append(dependency['parent'])
    
    for agent in agents:
        for subtask in agent['subtasks']:
            # Ensure subtasks are mapped correctly with agent ID
            agent_task[subtask].append(int(agent['id'].split()[-1]))  # Assuming 'id' is a string like 'Agent 0'
    
    # Create the workflow with fewer methods
    workflow = {}
    for i, task in enumerate(subtasks):
        # Ensure agent_task[i] is valid before accessing it
        assigned_agent = agent_task[i][0] if i in agent_task and len(agent_task[i]) > 0 else None
        
        # Check if an agent is assigned; otherwise, set a default (e.g., -1 or 'None')
        if assigned_agent is None:
            raise ValueError(f"No agent assigned for task {i}, make sure the agent-task mapping is correct.")
        
        temp_dic = {
            'id': f'task{i}',
            'objective': task['objective'],
            'agent_id': assigned_agent,
            # 'params': {'objective': task['objective'], 'agent': assigned_agent},
            'next': [f'task{key}' for key, value in dependencies.items() if i in value],
            'prev': [f'task{elem}' for elem in dependencies[i]],
            'status': 'pending',
            'data': '',
            'remaining_dependencies': len(dependencies[i]),
            'agent': agents[assigned_agent]['role']  # Use the assigned agent's role
        }
        workflow[f'task{i}'] = temp_dic
    
    # Convert to Workflow instance
    tasks = {task_id: Task(**task_info) for task_id, task_info in workflow.items()}
    return Workflow(tasks)