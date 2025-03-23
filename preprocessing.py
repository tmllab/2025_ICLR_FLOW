import json
from collections import defaultdict
from workflow import Task, Workflow
from history import History
import ast
import astor


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
def extract_funcs_var(code):
    """
    Extracts function definitions, import statements, and relevant global variables 
    from a code string and generates a new code string containing only these elements.
    """

    # Parse the code into an AST
    tree = ast.parse(code)

    # Collect functions, imports, and global assignments
    selected_nodes = []
    global_vars = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)):
            selected_nodes.append(node)
        elif isinstance(node, ast.Assign):  # Detect global assignments
            if all(isinstance(target, ast.Name) for target in node.targets):  # Ensure valid variable names
                global_vars.update(target.id for target in node.targets)
                selected_nodes.append(node)

    # Create a new AST containing only the selected elements
    new_tree = ast.Module(body=selected_nodes, type_ignores=[])

    # Convert the AST back into a code string
    new_code = astor.to_source(new_tree)
    return new_code

def extract_functions(code):
    """
    Extracts function definitions from a code string and generates a new code string containing only the functions.
    """
    # Parse the code into an AST
    tree = ast.parse(code)

    # Traverse the AST to extract function definitions
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)):
            functions.append(node)

    # Create a new AST containing only the extracted functions
    new_tree = ast.Module(body=functions, type_ignores=[])

    # Convert the AST back into a code string
    new_code = astor.to_source(new_tree)
    return new_code