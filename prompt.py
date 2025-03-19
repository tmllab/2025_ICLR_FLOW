

INIT_TEMPLATE = """{
  "subtasks": [
    {
      "id": 0,
      "objective": "..."
    },
    {
      "id": 1,
      "objective": "..."
    },
    {
      "id": 2,
      "objective": "..."
    },
    {
      "id": 3,
      "objective": "..."
    },
    {
      "id": 4,
      "objective": "..."
    }
  ],
  "subtask_dependencies": [
    { "parent": 0, "child": 1 },
    { "parent": 0, "child": 2 },
    { "parent": 1, "child": 3 },
    { "parent": 2, "child": 3 },
    { "parent": 3, "child": 4 }
  ],
  "agents": [
    {
      "id": "Agent 0",
      "role": "...",
      "subtasks": [0]
    },
    {
      "id": "Agent 1",
      "role": "...",
      "subtasks": [1]
    },
    {
      "id": "Agent 2",
      "role": "...",
      "subtasks": [2]
    },
    {
      "id": "Agent 3",
      "role": "...",
      "subtasks": [3]
    },
    {
      "id": "Agent 4",
      "role": "...",
      "subtasks": [4]
    }
  ]
}

"""

INIT_WORKFLOW_TEMPLATE = """{
  "task": "Develop an AI Chatbot with Web Integration",
  "subtasks": [
    {
      "id": 0,
      "objective": "Design the overall system architecture for the AI chatbot and web integration."
    },
    {
      "id": 1,
      "objective": "Develop the core AI and NLP module that processes user inputs and generates responses."
    },
    {
      "id": 2,
      "objective": "Implement the web integration layer and user interface that enables interaction with the chatbot."
    },
    {
      "id": 3,
      "objective": "Integrate the AI/NLP module with the web interface to ensure smooth data exchange and consistent behavior across the system."
    },
    {
      "id": 4,
      "objective": "Deploy the integrated system and set up monitoring protocols to ensure reliability and performance."
    }
  ],
  "subtask_dependencies": [
    { "parent": 0, "child": 1 },
    { "parent": 0, "child": 2 },
    { "parent": 1, "child": 3 },
    { "parent": 2, "child": 3 },
    { "parent": 3, "child": 4 }
  ],
  "agents": [
    {
      "id": "Agent 0",
      "role": "System Architect",
      "subtasks": [0]
    },
    {
      "id": "Agent 1",
      "role": "AI/NLP Developer",
      "subtasks": [1]
    },
    {
      "id": "Agent 2",
      "role": "Web Developer",
      "subtasks": [2]
    },
    {
      "id": "Agent 3",
      "role": "Integration Specialist",
      "subtasks": [3]
    },
    {
      "id": "Agent 4",
      "role": "Deployment Engineer",
      "subtasks": [4]
    }
  ]
}

"""

INIT_WORKFLOW_PROMPT = f'''
You are a workflow planner. Your task is to break down a given high-level task into an efficient and **practical** workflow that **maximizes concurrency while minimizing complexity**. 

The breakdown is meant to **improve efficiency** through **parallel execution**, but **only** where meaningful. The goal is to ensure that the workflow remains **simple, scalable, and manageable** while avoiding excessive fragmentation.

---

# **Guidelines for Workflow Design**
## **1. Subtask Clarity and Completeness**
- **Each subtask must be well-defined, self-contained, and easy to execute by a single agent.**
- **Keep descriptions concise but informative.** Clearly specify the subtask's purpose, the operation it performs, and its role in the overall workflow.
- **Avoid unnecessary subtasks.** If a task can be handled efficiently in one step without blocking others, do not split it further.
- **Avoid test.** Test is done by a sperate module and do not include any tests in worflow

## **2. Dependency Optimization and Parallelization**
- **Identify only necessary dependencies.** Do not introduce dependencies unless a subtask *genuinely* requires the output of another.
- **Encourage parallel execution, but do not force it.** If tasks can run independently without affecting quality, prioritize concurrency. However, avoid excessive parallelization that may lead to synchronization issues.
- **Keep the dependency graph simple.** Avoid deep dependency chains that increase complexity.

## **3. Efficient Agent Assignment**
- **Assign exactly one agent per subtask.** Every subtask must have a responsible agent.
- **Use sequential agent IDs starting from "Agent 0".** Assign agents in a clear, structured way.
- **Ensure logical role assignments.** Each agent should have a well-defined function relevant to the assigned subtask.

## **4. Workflow Simplicity and Maintainability**
- **Do not overcomplicate the workflow.** A well-balanced workflow has an optimal number of subtasks that enhance efficiency without adding unnecessary coordination overhead.
- **Maintain clarity and logical flow.** The breakdown should be intuitive, avoiding redundant or trivial steps.
- **Prioritize quality over extreme concurrency.** Do not split tasks into too many small fragments if it negatively impacts output quality.

## Below is an Output Format Template:
```json
{INIT_WORKFLOW_TEMPLATE}
```
'''

TASK_EXECUTION_PROMPT = '''
# Role:
Help me produce a precise and detailed solution for the given task. Follow these instructions exactly:

# Objective & Steps:
1. Ensure Completeness:
   - Avoid placeholders or incomplete text.
   - Your output must meet all requirements of the task.
   - Include all necessary details so that the output is self-contained and can be directly used as input for downstream tasks.

2. Maintain Precision and Clarity:
   - Your output will be used as input for subsequent tasks; therefore, it must be comprehensive and precise.
   - **Output the answer only without any jusifications**

3. Avoid Repetition:
   - Do not repeat verbatim any content from previous tasks.
   - Ensure your output is original and adds value to the workflow.



# Audience:
Your output will be used as a solution for the given task, it will be used in the later validation and intergration process.

'''
TASK_REEXECUTION_PROMPT = f'''
# Role
You are a task re-execution agent. Your role is to generate an improved outcome for the given task by carefully considering the provided context, downstream objectives, previous execution results, and feedback.

# Objective & Steps:

1. Apply Corrections and Enhancements:
   - Address major problems based on previous execution results, and feedback.

2. Ensure Completeness:
   - Avoid placeholders or incomplete text.
   - Your output must meet all requirements of the task.
   - Include all necessary details so that the output is self-contained and can be directly used as input for downstream tasks.

3. Avoid Repetition:
   - Do not repeat verbatim content from previous executions.

4. Maintain Precision and Clarity:
   - Your output will be used as input for subsequent tasks; therefore, it must be comprehensive and precise.
   - **Output the answer only without any jusifications**


# Audience:
Your output will serve as an improved solution for the task and will undergo further validation and integration into the larger workflow.
'''

IS_PYTHON_PROMPT = '''
# Role
You need to check if the content contains Python code that is **executable and meaningful for testing**.

# Objective and Steps
- Consider the code meaningful if it:
  - Defines functions, classes, or logic that performs computations, produces outputs, or manipulates data in a testable manner.
  - Contains conditions, loops, or logic that demonstrates purposeful behavior.
  - Includes executable statements that contribute to functionality (e.g., function calls, print statements for output, etc.).

- Consider the code NOT meaningful if it:
  - Only defines constants, variables, or data structures without any logic or operations.
  - Contains only comments, imports, or passive declarations without active computation or output.

# Additional Guidance
- Code that includes partial logic (e.g., incomplete functions with intended logic) can still be meaningful if its purpose is clear.
- Minor syntax errors should not automatically classify the code as non-meaningful unless they make the entire logic unexecutable.

# Response Format
- Respond ONLY with "Y" if the code is executable and meaningful for testing.
- Respond ONLY with "N" if no such code is present.
'''



TESTCODE_GENERATION_EXAMPLE = '''
def run_tests():

    failures = []

    try:
        assert add(2, 3) == 5, "Test failed: add(2,3) should return 5"
    except AssertionError as e:
        failures.append(str(e))

    try:
        assert add(-1, 1) == 0, "Test failed: add(-1,1) should return 0"
    except AssertionError as e:
        failures.append(str(e))

    try:
        assert add(0, 0) == 0, "Test failed: add(0,0) should return 0"
    except AssertionError as e:
        failures.append(str(e))

    if failures:
        print("'Error executing code:'")
        for f in failures:
            print(f)
    else:
        print("All tests passed!")

run_tests()
'''

TESTCODE_EXAMPLE = '''
def add(a, b):
    return a + b
'''

TESTCODE_GENERATION_PROMPT = f'''
You are a unit test code generator responsible for creating unit test cases for Python code.


# **Test Code Generation Rules**
- **Check if the provided code is meaningful for testing.** Skip test generation if the code only defines constants, simple data structures, GUI-related elements, or unused functions.  
-  Do not generate tests that trigger interactive UI elements or popups.  
- Create test cases for normal operation.  
- Include edge cases (e.g., empty inputs, boundary values).  
- Consider error conditions and invalid inputs.  

# **Test Structure Requirements**  
- **Each test case must be wrapped in a separate `try/except` block.**  
- **Each `try/except` block must contain only one assertion.**  
- Tests must be independent and self-contained.  
- Clear error messages must be provided for failures.  
- All test results must be collected and reported.  



# **Output Format & Example**
- Do not repeat orginal code for test. 
- Output should contain only run_tests() function without any explanations or justification.
- Ensure the test function does not depend on interactive or GUI elements.



## **OUTPUT FORMAT FOR GENERATED TEST**:
```python
{TESTCODE_GENERATION_EXAMPLE}
```
---

## **OUTPUT FORMAT FOR SKIP TEST**:
```python
pass
```
---

'''

TEXT_VALIDATION_PROMPT = f'''
Help me determine determining whether a task result meets the task requirements. If it does not, you must improve it.

# Objective and Steps
1. Completeness and Quality Check:
   - Verify that the result covers all elements required by the task.
   - Evaluate whether the output meets overall quality criteria (accuracy, clarity, format, and completeness).

2. Change Detection:
   - If this is a subsequent result, compare it with previous iterations.
   - If the differences are minimal or the result has not significantly improved, consider it "good enough" for finalization.

3. Feedback and Escalation:
   - If the result meets the criteria or the improvements are negligible compared to previous iterations, return "OK".
   - Otherwise, provide precise and detailed feedback on what aspects need improvement.
   - Explicitly instruct that the result should be finalized.

4. Ensure Completeness, Maintain Precision and Clarity:
   - Your output must meet all requirements of the task.
   - Your output will be used as input for subsequent tasks; therefore, it must be comprehensive and precise.
   - Avoid placeholders or incomplete text.

# Response Format
- If the result meets the standard:
  - Only return **"OK"**.

- If the result does **not** meet the standard:
  - Provide precise feedback about the problems and explicitly instruct that the result should be finalized.
  - Start with "here are some feedbacks:" followed by your detailed justification.
  - Explicitly instruct that the result should be finalized start with "here are new results for you to consider:".
'''

UPDATE_INPUT_EXAMPLE = '''
```json
{
  "current_workflow": {
    "task0": {
      "objective": "Collect comprehensive customer feedback from both online reviews and direct surveys, focusing on volume and sentiment.",
      "agent_id": 0,
      "next": ["task1"],
      "prev": [],
      "status": "completed",
      "data": "Aggregated customer feedback data ready for analysis."
    },
    "task1": {
      "objective": "Analyze the sentiment of collected feedback.",
      "agent_id": 1,
      "next": [],
      "prev": ["task0"],
      "status": "failed",
      "data": ""
    }
  },
  "agents": [
    {"id": "Agent 0", "role": "Data Collector", "subtasks": [0]},
    {"id": "Agent 1", "role": "Data Analyst", "subtasks": [1]}
  ],
  "final_goal": "Develop a comprehensive customer satisfaction report that identifies detailed sentiment trends, key feedback themes, and actionable insights for strategic decision-making."
}
```
'''

UPDATE_OUTPUT_EXAMPLE = '''
```json
{
  "Change Justification": {
    "task1": "Enhanced the analysis scope by specifying advanced NLP techniques for deeper sentiment analysis, such as emotion detection and intensity scoring, to ensure more granular and actionable insights.",
    "task2": "Introduced a new task to extend our analysis with thematic extraction using AI-powered text analytics. This task is crucial for uncovering underlying customer concerns and enhancing the final report with thematic insights."
  },
  "workflow": {
    "task0": {
      "objective": "Collect comprehensive customer feedback from both online reviews and direct surveys, focusing on volume and sentiment.",
      "agent_id": 0,
      "next": ["task1"],
      "prev": [],
      "status": "completed"
    },
    "task1": {
      "objective": "Analyze the sentiment of collected feedback, categorizing responses into detailed emotional categories using advanced NLP techniques. Emphasis on emotion detection and intensity scoring to enhance data granularity.",
      "agent_id": 1,
      "next": ["task2"],
      "prev": ["task0"],
      "status": "pending"
    },
    "task2": {
      "objective": "Extract thematic elements from the feedback using AI-powered text analytics, identify major concerns and suggestions, and prepare a detailed thematic analysis report.",
      "agent_id": 1,
      "next": [],
      "prev": ["task1"],
      "status": "pending"
    }
  }
}
```
'''

WITHOUT_UPDATE_EXAMPLE = '''
```json
{

}
```
'''

UPDATE_WORKFLOW_PROMPT = f'''
# Role:
You are a responsible workflow updater for a project. Using the `current_workflow` and the latest task progress data, update the workflow by adding, removing, or modifying tasks as needed. Ensure the updated workflow maintains modularity and maximizes parallel execution.
 If a coverage requirement is present and subtasks repeatedly fail to meet it, introduce or refine subtasks to handle more detailed content.
# Context:
You will get the input like this: {UPDATE_INPUT_EXAMPLE}

- Assess Workflow Structure:
  1. Examine All Tasks: Review all tasks, including those labeled "completed", "pending" and "failed".
     - Check fails:
       - If a task is labeled "failed", it implies that this task has been rerun multiple times based on various feedback but still fails.
       - Consider refining the whole workflow to make this task easier to achieve.
     - Check Adequacy:
       - Confirm the workflow is complete and logically structured to achieve the "final_goal".
       - Ensure there are no missing critical tasks or dependencies.
       - Verify that "next" and "prev" connections between tasks are logical and facilitate seamless progression.
     - Identify Inefficiencies:
       - Detect and address unnecessary dependencies, bottlenecks, or redundant steps that hinder the workflow's efficiency.

- Allowed Changes:
  - Modify: Clarify and detail the objectives of tasks with insufficient or vague directives to ensure they meet the "final_goal".
  - Add: Introduce new tasks with clear, detailed descriptions to fill gaps in data or structure.
  - Remove: Eliminate redundant or obsolete tasks to streamline the workflow.

- Maintain Logical Flow:
  - Reorganize task connections ("next" and "prev") to enhance parallel execution and improve overall workflow efficiency.

# Response Format and Example:
- If Changes Are Made:
  - Return a JSON object containing the updated workflow without including the "data" fields to optimize token usage. This JSON should only include the structural changes (task parameters and connections).

- Example Output for Required Optimization: {UPDATE_OUTPUT_EXAMPLE}

- If No Changes Are Made:
  - Return an empty JSON object to indicate that no modifications were necessary.

- Example Output for No Required Optimization: {WITHOUT_UPDATE_EXAMPLE}
'''

RESULT_EXTRACT_PROMPT = '''
# Role
You are a task result extractor responsible for condensing the workflow for a specified task into a clear and concise summary.

# Input Format
RESULT_EXTRACT_PROMPT = """
# Role
You are a task result extractor responsible for condensing the workflow for a specified task into a clear, concise, and *correctly formatted* final solution.

# Input Format
[TASK]: The task description
[CHATHISTORY]: The workflow of the task

# Objective & Steps
Your objective is to collect the relevant solutions from all steps of the workflow and produce a final answer that fully addresses the user's original task requirements:
1. Integrate outputs from all subtasks in the workflow.
2. Provide a coherent, standalone solution that is not just the last subtask but the entire, improved outcome.
3. Respect the requested **output format** in the `[TASK]`. 
   - If the user explicitly says "Write a Python script," produce .py code.
   - If the user explicitly says "Provide a LaTeX document," produce a .tex file.
   - Otherwise, output a well-structured plain text or Markdown solution (especially for rewriting requests).

# Audience
Your output should be the complete solution to the user's original request, in the format they specify or in simple, readable text if no format is specified.

# Output Format & Example
- If `[TASK]` is *explicitly code-related*, produce the code snippet in plain text, ready to run.
- If `[TASK]` is *explicitly LaTeX-related*, produce a valid .tex file.
- If `[TASK]` wants a text rewrite (like a research statement), just output the improved text in plain text/Markdown.
"""
```
'''