from typing import Dict, Any, List
import networkx as nx
from collections import deque, defaultdict
import json

class Task:
    """Data structure representing a workflow task."""
    def __init__(self, id: str, objective: str, agent_id: int, next: List[str], prev: List[str],
                 status: str = 'pending', data: str = '', agent: str = ''):
        self.id = id
        self.objective = objective
        self.agent_id = agent_id
        self.next = next
        self.prev = prev
        self.status = status
        self.data = data
        self.remaining_dependencies = len(self.prev)
        self.agent = agent


    def to_dict(self) -> Dict[str, Any]:
        """Converts the Task object to a dictionary for JSON serialization."""
        return {
            'id': self.id,
            'objective': self.objective,
            'agent_id': self.agent_id,
            'next': self.next,
            'prev': self.prev,
            'status': self.status,
            'data': self.data,
            'remaining_dependencies': self.remaining_dependencies,
            'agent': self.agent
        }



class Workflow:
    """Manages a collection of tasks and their dependencies."""
    def __init__(self, tasks: Dict[str, Task]):
        self.tasks = tasks

    def _build_graph(self) -> nx.DiGraph:
        """Build a directed graph from the tasks' dependencies."""
        graph = nx.DiGraph()
        for task in self.tasks.values():
            graph.add_node(task.id)
        for task in self.tasks.values():
            for prev_task_id in task.prev:
                graph.add_edge(prev_task_id, task.id)  # edge: prev -> task
        self.graph = graph
        return graph


    def get_context(self, task_id: str) -> str:
        """
        Returns a nicely formatted string representing the context for the given task.
        For each previous task that is completed, this includes:
          - Task id.
          - Objective (from task.params.get('objective', ...)).
          - Result (from task.data).
        """
        if task_id not in self.tasks:
            return f"Task '{task_id}' not found in workflow."
        
        context_lines = []
        for prev_id in self.tasks[task_id].prev:
            if prev_id in self.tasks and self.tasks[prev_id].status == 'completed':
                prev_task = self.tasks[prev_id]
                # objective = prev_task.params.get('objective', 'No objective defined')
                objective = prev_task.objective
                result = prev_task.data if prev_task.data else 'No result available'
                context_lines.append(
                    f"Task {prev_id}:\n"
                    f"  Objective: {objective}\n"
                    f"  Result: {result}"
                )
        
        if context_lines:
            return "\n".join(context_lines)
        else:
            return "No completed previous tasks context available."

    def get_downsteam_objectives(self, task_id: str) -> str:
        """
        Returns a nicely formatted string of the objectives for each downstream task.
        For each task listed in the 'next' list, the objective is fetched from
        task.params.get('objective', 'No objective defined').
        """
        if task_id not in self.tasks:
            return f"Task '{task_id}' not found in workflow."
        
        objective_lines = []
        for next_id in self.tasks[task_id].next:
            if next_id in self.tasks:
                next_task = self.tasks[next_id]
                # objective = next_task.params.get('objective', 'No objective defined')
                objective = next_task.objective
                objective_lines.append(
                    f"Task {next_id}:\n"
                    f"  Objective: {objective}"
                )
        
        if objective_lines:
            return "\n".join(objective_lines)
        else:
            return "No downstream objectives available."
    
    def handle_task_completion(self, task_id: str) -> List[str]:
        """
        Update the downstream tasks of a just-completed task.
        Decrements the remaining_dependencies count for each downstream task.
        Returns the list of tasks that are now ready to be scheduled.
        """
        if task_id not in self.tasks:
            return []
        task_obj = self.tasks[task_id]
        ready_to_run = []
        for next_id in task_obj.next:
            if next_id in self.tasks:
                downstream_task = self.tasks[next_id]
                if downstream_task.status == 'pending':
                    downstream_task.remaining_dependencies -= 1
                    if downstream_task.remaining_dependencies <= 0:
                        ready_to_run.append(next_id)
        return ready_to_run

    def add_task(self, id: str, objective: str, agent_id: int, next: List[str], prev: List[str],
                 status: str = 'pending', data: str = '', remaining_dependencies: int = 0,
                 agent: str = '', compute_dependencies: bool = True):
        """
        Add a new task to the workflow and update dependency links in related tasks.
        
        If compute_dependencies is True (the default), then the task's 
        remaining_dependencies is computed as the count of parent's tasks 
        (listed in prev) that exist in the workflow and are not completed.
        
        In a merge scenario, where tasks might be added out-of-order, set
        compute_dependencies=False so that remaining_dependencies is initialized to 0,
        and later recalc_dependencies() is used to update counts.
        """
        if id in self.tasks:
            raise ValueError(f"Task with id '{id}' already exists.")
        
        if compute_dependencies:
            if remaining_dependencies == 0:
                remaining_dependencies = sum(
                    1 for p in prev if p in self.tasks and self.tasks[p].status != 'completed'
                )
        else:
            # In merge scenarios, do not compute dependencies yet.
            remaining_dependencies = 0
        
        # new_task = Task(id, params, next, prev, status, data, remaining_dependencies, agent)
        new_task = Task(id, objective, agent_id, next, prev, status, data, remaining_dependencies, agent)
        self.tasks[id] = new_task

        # Update related tasks: add this task id to each parent's `next` list.
        for p in prev:
            if p in self.tasks and id not in self.tasks[p].next:
                self.tasks[p].next.append(id)
        # Similarly, for each child task, add this task id to its `prev` list.
        for n in next:
            if n in self.tasks and id not in self.tasks[n].prev:
                self.tasks[n].prev.append(id)





    def edit_task(self, id: str, objective: str, agent_id: int, next: List[str], prev: List[str],
                  status: str = 'pending', data: str = '', remaining_dependencies: int = 0, agent: str = ''):
        """
        Edit an existing task and update dependency links.
        Raises a ValueError if the task does not exist.
        """
        if id not in self.tasks:
            raise ValueError(f"Task with id '{id}' does not exist.")
        task = self.tasks[id]
        
        # Save old dependency lists for later updates.
        old_prev = set(task.prev)
        old_next = set(task.next)
        new_prev = set(prev)
        new_next = set(next)
        
        # Update task fields.
        # task.params = params
        task.objective = objective
        task.agent_id = agent_id
        task.next = next
        task.prev = prev
        task.status = status
        task.data = data
        task.agent = agent
        if remaining_dependencies == 0:
            task.remaining_dependencies = sum(
                1 for p in task.prev if p in self.tasks and self.tasks[p].status != 'completed'
            )
        else:
            task.remaining_dependencies = remaining_dependencies

        # Remove references from tasks no longer related.
        for p in old_prev - new_prev:
            if p in self.tasks and id in self.tasks[p].next:
                self.tasks[p].next.remove(id)
        for n in old_next - new_next:
            if n in self.tasks and id in self.tasks[n].prev:
                self.tasks[n].prev.remove(id)
        
        # Add references for new dependencies.
        for p in new_prev - old_prev:
            if p in self.tasks and id not in self.tasks[p].next:
                self.tasks[p].next.append(id)
        for n in new_next - old_next:
            if n in self.tasks and id not in self.tasks[n].prev:
                self.tasks[n].prev.append(id)

    def del_task(self, id: str):
        """
        Delete a task from the workflow and clean up references in related tasks.
        Raises a ValueError if the task does not exist.
        """
        if id not in self.tasks:
            raise ValueError(f"Task with id '{id}' does not exist.")
        task = self.tasks[id]
        # Remove this task id from its parents' next lists.
        for p in task.prev:
            if p in self.tasks and id in self.tasks[p].next:
                self.tasks[p].next.remove(id)
        # Remove this task id from its children’s prev lists.
        for n in task.next:
            if n in self.tasks and id in self.tasks[n].prev:
                self.tasks[n].prev.remove(id)
        # Also remove any stray references from all tasks.
        for t in self.tasks.values():
            if id in t.prev:
                t.prev.remove(id)
            if id in t.next:
                t.next.remove(id)
        del self.tasks[id]
        self.recalculate_dependencies()

    def merge_workflow(self, new_workflow: Dict[str, Any]):
        """
        Merge an updated workflow with the current one.
        This method covers:
          1. Deleting tasks that are not in the new workflow.
          2. Adding new tasks.
          3. Editing tasks that exist in both workflows (if any key attribute has changed).
        
        When adding tasks in the merge, we pass compute_dependencies=False to add_task so that
        dependency counts are not computed prematurely. Once all tasks are merged, we recalculate
        dependencies and propagate pending status downstream.
        """
        # --- 1. Delete tasks that are no longer present ---
        current_ids = set(self.tasks.keys())
        new_ids = set(new_workflow.keys())
        for task_id in current_ids - new_ids:
            self.del_task(task_id)

        # --- 2. Add new tasks and edit existing tasks ---
        for task_id, new_task_data in new_workflow.items():
            if task_id in self.tasks:
                # If the current task does not match the new definition, edit it.
                if not self.tasks_are_equal(self.tasks[task_id], new_task_data):
                    # If a task was completed but now has changed, roll it back to pending.
                    new_status = "pending"
                    self.edit_task(
                        id=task_id,
                        # params=new_task_data.get('params', {}),
                        objective=new_task_data.get('objective', ''),
                        agent_id=new_task_data.get('agent_id', -1),
                        next=new_task_data.get('next', []),
                        prev=new_task_data.get('prev', []),
                        status=new_status,
                        data='',
                        remaining_dependencies=0,
                        agent=new_task_data.get('agent', '')
                    )
                # Otherwise, leave the task as is.
            else:
                # Add new task with compute_dependencies=False since some parents may not yet exist.
                self.add_task(
                    id=task_id,
                    # params=new_task_data.get('params', {}),
                    objective=new_task_data.get('objective', ''),
                    agent_id=new_task_data.get('agent_id', -1),
                    next=new_task_data.get('next', []),
                    prev=new_task_data.get('prev', []),
                    status='pending',
                    data='',
                    remaining_dependencies=0,
                    agent=new_task_data.get('agent', ''),
                    compute_dependencies=False
                )

        # --- 3. Recalculate dependencies and propagate pending status ---
        self.recalculate_dependencies()
        self._propagate_pending_status()

    def _propagate_pending_status(self):
        """
        Propagate pending status downstream.
        If a task is marked as completed but has any parent that is not completed,
        change its status to pending (and clear its data).
        """
        changed = True
        while changed:
            changed = False
            for task in self.tasks.values():
                # Only consider tasks that are marked as completed.
                if task.status == 'completed':
                    for parent_id in task.prev:
                        if parent_id in self.tasks and self.tasks[parent_id].status != 'completed':
                            task.status = 'pending'
                            task.data = ''
                            changed = True
                            break

    def tasks_are_equal(self, task_obj: Task, new_task_data: Dict[str, Any]) -> bool:
        """
        Determine if a task matches its new definition.
        Comparison is made on params, next, and prev lists.
        """
        # return (task_obj.params == new_task_data.get('params') and
        #         task_obj.next == new_task_data.get('next') and
        #         task_obj.prev == new_task_data.get('prev'))
        return (task_obj.objective == new_task_data.get('objective') and
                task_obj.next == new_task_data.get('next') and
                task_obj.prev == new_task_data.get('prev'))

    def recalculate_dependencies(self):
        """Recalculate the remaining dependencies for each task."""
        for task in self.tasks.values():
            if task.status == 'completed':
                task.remaining_dependencies = 0
            else:
                task.remaining_dependencies = sum(
                    1 for p in task.prev if p in self.tasks and self.tasks[p].status != 'completed'
                )

    def calculate_average_parallelism(self) -> float:
        """
        Calculate the average parallelism of the workflow.
        (Average number of tasks that can run simultaneously at each step.)
        """
        # BFS-based calculation grouping tasks by depth.
        depth = {task_id: 0 for task_id in self.tasks}
        queue = deque([task_id for task_id, task in self.tasks.items() if not task.prev])
        step_tasks = defaultdict(list)
        while queue:
            current_id = queue.popleft()
            current_task = self.tasks[current_id]
            step_tasks[depth[current_id]].append(current_id)
            for next_id in current_task.next:
                if next_id in depth:
                    depth[next_id] = depth[current_id] + 1
                    queue.append(next_id)
        parallelism = [len(tasks) for tasks in step_tasks.values()]
        return sum(parallelism) / len(parallelism) if parallelism else 0.0

    def calculate_dependency_complexity(self) -> float:
        """Calculate the dependency complexity of the workflow."""
        self._build_graph()
        nodes = list(self.graph.nodes)
        degrees = [self.graph.degree(n) for n in nodes]
        average_degree = sum(degrees) / len(nodes)
        complexity = sum((self.graph.degree(n) - average_degree) ** 2 for n in nodes)
        return (complexity / len(nodes)) ** 0.5

    def all_completed(self) -> bool:
        """Determine if all tasks have been completed."""
        return all(task.status == 'completed' for task in self.tasks.values())

    def get_pending_tasks(self) -> List[Task]:
        """Return all tasks that are pending and ready to run."""
        return [task for task in self.tasks.values() if task.status == 'pending' and task.remaining_dependencies == 0]

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Workflow object to a dictionary for JSON serialization."""
        return {'tasks': {task_id: task.to_dict() for task_id, task in self.tasks.items()}}

    def to_json(self, filepath: str):
        """Serializes the workflow to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
