from workflow import Task, Workflow

def customize_workflow():
    tasks_ls = []
    tasks_dic = {}

    # Example
    task0 = Task(
        id = "task0",
        objective = "objective0",
        agent_id = 0,
        next = ["task1"],
        prev = [],
        agent = "agent0",
    )
    tasks_ls.append(task0)

    task1 = Task(
        id = "task1",
        objective = "objective1",
        agent_id = 1,
        next = [],
        prev = ["task0"],
        agent = "agent2",
    )
    tasks_ls.append(task1)

    for task in tasks_ls:
        task.calculate_dependencies()
        tasks_dic[task.id] = task

    customize_workflow = Workflow(tasks_dic)

    return customize_workflow