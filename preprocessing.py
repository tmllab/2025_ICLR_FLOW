import json
from workflow import Task, Workflow
from history import History

def process_context(context: str):
    """
    Process context string and extract workflow data with simple, consistent format.
    
    Args:
        context (str): JSON string containing workflow data in format:
                      {"workflow": {"task0": {...}, ...}, "agents": [...]}
        
    Returns:
        Workflow: Processed workflow object
        
    Raises:
        ValueError: If the JSON structure is invalid or missing required fields
        json.JSONDecodeError: If JSON parsing fails
    """
    try:
        # Parse JSON directly
        context_dict = json.loads(context)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}")
    
    # Validate required keys exist
    required_keys = ['workflow', 'agents']
    missing_keys = [key for key in required_keys if key not in context_dict]
    
    if missing_keys:
        raise ValueError(f"Missing required keys in GPT response: {missing_keys}. "
                        f"Available keys: {list(context_dict.keys())}")
    
    # Extract workflow and agents
    workflow_dict = context_dict['workflow']
    agents = context_dict['agents']
    
    # Validate data types
    if not isinstance(workflow_dict, dict):
        raise ValueError(f"'workflow' should be a dict, got {type(workflow_dict)}")
    
    if not isinstance(agents, list):
        raise ValueError(f"'agents' should be a list, got {type(agents)}")
    
    if not workflow_dict:
        raise ValueError("'workflow' dict is empty")
        
    if not agents:
        raise ValueError("'agents' list is empty")
    
    # Validate agents structure
    for i, agent in enumerate(agents):
        if not isinstance(agent, dict):
            raise ValueError(f"Invalid agent structure at index {i}: {agent}. Expected dict.")
        
        required_agent_keys = ['id', 'role']
        missing_agent_keys = [key for key in required_agent_keys if key not in agent]
        if missing_agent_keys:
            raise ValueError(f"Agent missing required keys {missing_agent_keys}: {agent}")
        
        # Validate agent ID format
        try:
            agent_id = int(agent['id'].split()[-1])  # Extract agent ID from "Agent 0" format
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid agent ID format '{agent['id']}'. Expected format like 'Agent 0': {e}")
    
    # Build tasks directly from workflow dict
    tasks = {}
    for task_id, task_data in workflow_dict.items():
        if not isinstance(task_data, dict):
            raise ValueError(f"Invalid task structure for '{task_id}': {task_data}. Expected dict.")
        
        # Validate required task fields
        required_task_keys = ['objective', 'agent_id', 'next', 'prev']
        missing_task_keys = [key for key in required_task_keys if key not in task_data]
        if missing_task_keys:
            raise ValueError(f"Task '{task_id}' missing required keys {missing_task_keys}: {task_data}")
        
        # Validate agent_id is within bounds
        agent_id = task_data['agent_id']
        if not isinstance(agent_id, int) or agent_id >= len(agents):
            raise ValueError(f"Task '{task_id}' has invalid agent_id {agent_id}. Must be integer < {len(agents)}")
        
        # Validate next/prev are lists
        if not isinstance(task_data['next'], list):
            raise ValueError(f"Task '{task_id}' 'next' should be a list, got {type(task_data['next'])}")
        
        if not isinstance(task_data['prev'], list):
            raise ValueError(f"Task '{task_id}' 'prev' should be a list, got {type(task_data['prev'])}")
        
        # Create task object - status is always 'pending' for new workflows
        task_info = {
            'id': task_id,
            'objective': task_data['objective'],
            'agent_id': agent_id,
            'next': task_data['next'],
            'prev': task_data['prev'],
            'status': 'pending',
            'history': History(),
            'agent': agents[agent_id]['role'],
            'output_format': task_data.get('output_format', '')
        }
        tasks[task_id] = Task(**task_info)
    
    return Workflow(tasks)