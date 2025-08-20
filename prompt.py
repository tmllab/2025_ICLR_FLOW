

INIT_TEMPLATE = """{
  "subtasks": [
    {
      "id": 0,
      "objective": "...",
      "output_format": "..."
    },
    {
      "id": 1,
      "objective": "...",
      "output_format": "..."
    },
    {
      "id": 2,
      "objective": "...",
      "output_format": "..."
    },
    {
      "id": 3,
      "objective": "...",
      "output_format": "..."
    },
    {
      "id": 4,
      "objective": "...",
      "output_format": "..."
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
  "workflow": {
    "task0": {
      "objective": "Design the overall system architecture for the AI chatbot and web integration.",
      "agent_id": 0,
      "next": ["task1", "task2"],
      "prev": []
    },
    "task1": {
      "objective": "Develop the core AI and NLP module that processes user inputs and generates responses.",
      "agent_id": 1,
      "next": ["task3"],
      "prev": ["task0"]
    },
    "task2": {
      "objective": "Implement the web integration layer and user interface that enables interaction with the chatbot.",
      "agent_id": 2,
      "next": ["task3"],
      "prev": ["task0"]
    },
    "task3": {
      "objective": "Integrate the AI/NLP module with the web interface to ensure smooth data exchange and consistent behavior across the system.",
      "agent_id": 3,
      "next": ["task4"],
      "prev": ["task1", "task2"]
    },
    "task4": {
      "objective": "Deploy the integrated system and set up monitoring protocols to ensure reliability and performance.",
      "agent_id": 4,
      "next": [],
      "prev": ["task3"]
    }
  },
  "agents": [
    {
      "id": "Agent 0",
      "role": "System Architect"
    },
    {
      "id": "Agent 1", 
      "role": "AI/NLP Developer"
    },
    {
      "id": "Agent 2",
      "role": "Web Developer"
    },
    {
      "id": "Agent 3",
      "role": "Integration Specialist"
    },
    {
      "id": "Agent 4",
      "role": "Deployment Engineer"
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
- **Ensure that the workflow meets all requirements of the task.**
- **Keep descriptions concise but informative.** Clearly specify the subtask's purpose, the operation it performs, and its role in the overall workflow.
- **Avoid unnecessary subtasks.** If a task can be handled efficiently in one step without blocking others, do not split it further.


## **2. Dependency Optimization and Parallelization**
- **Identify only necessary dependencies.** Do not introduce dependencies unless a subtask *genuinely* requires the output of another.
- **Encourage parallel execution, but do not force it.** If tasks can run independently without affecting quality, prioritize concurrency. However, avoid excessive parallelization that may lead to synchronization issues.
- **Keep the dependency graph simple.** Avoid deep dependency chains that increase complexity.

## **3. Efficient Agent Assignment**
- **Assign exactly one agent per task using agent_id (0, 1, 2, etc.).** Every task must have a responsible agent.
- **Use sequential agent IDs starting from "Agent 0".** Assign agents in a clear, structured way.
- **Ensure logical role assignments.** Each agent should have a well-defined function relevant to the assigned task.

## **4. Workflow Simplicity and Maintainability**
- **Do not overcomplicate the workflow.** A well-balanced workflow has an optimal number of subtasks that enhance efficiency without adding unnecessary coordination overhead.
- **Maintain clarity and logical flow.** The breakdown should be intuitive, avoiding redundant or trivial steps.
- **Prioritize quality over extreme concurrency.** Do not split tasks into too many small fragments if it negatively impacts output quality.

## **Required JSON Output Format:**

Your response must be a JSON object with exactly this structure:
- **"workflow"**: A dict where keys are task IDs ("task0", "task1", etc.) and values contain:
  - **"objective"**: Clear description of what this task accomplishes
  - **"agent_id"**: Integer (0, 1, 2, etc.) identifying which agent handles this task
  - **"output_format"**: Required output format for this task (e.g., "JSON", "LaTeX", "Python code", "Markdown", "Plain text", etc.)
  - **"next"**: List of task IDs that depend on this task (can be empty [])
  - **"prev"**: List of task IDs this task depends on (can be empty [])
- **"agents"**: A list of agent objects with:
  - **"id"**: String like "Agent 0", "Agent 1", etc.
  - **"role"**: Descriptive role name for the agent

**Example Template:**
```json
{INIT_WORKFLOW_TEMPLATE}
```

**Important Notes:**
- Do NOT include "status" or "data" fields - these are managed by the system
- Task dependencies are expressed through "next" and "prev" arrays
- Agent IDs must be consistent between workflow tasks and agents list
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
You are a smart Python code validator responsible for creating appropriate validation tests based on the code complexity and type.

# **Code Analysis & Test Strategy**

1. **Simple Functions/Utilities**: Generate unit tests with assertions
2. **Complex Applications (Games/GUIs/Interactive)**: Generate syntax and import validation only
3. **Constants/Data/Incomplete Code**: Skip testing entirely

# **Decision Rules**

**GENERATE UNIT TESTS** if code contains:
- Pure functions with clear inputs/outputs (math, string processing, algorithms)
- Utility classes without external dependencies
- Data processing functions
- Simple business logic

**GENERATE SYNTAX/IMPORT VALIDATION** if code contains:
- Game development (Pygame, Tkinter, etc.)
- Web frameworks (Flask, Django, etc.)
- GUI applications with event loops
- Real-time or interactive systems
- Complex class hierarchies with external dependencies

**SKIP TESTING** if code contains:
- Only constants, variables, or data structures
- Incomplete functions or pseudocode
- Only imports or comments

# **Test Structure Requirements**  
- **Each test case must be wrapped in a separate `try/except` block.**  
- **Each `try/except` block must contain only one assertion.**  
- Tests must be independent and self-contained.  
- Clear error messages must be provided for failures.  
- All test results must be collected and reported.  

# **Output Format & Examples**

## **OUTPUT FORMAT FOR UNIT TESTS**:
```python
{TESTCODE_GENERATION_EXAMPLE}
```

## **OUTPUT FORMAT FOR SYNTAX/IMPORT VALIDATION**:
```python
def run_tests():
    failures = []
    
    try:
        # Test syntax by attempting compilation
        compile(open(__file__).read(), __file__, 'exec')
    except SyntaxError as e:
        failures.append(f"Syntax Error: {{e}}")
    
    try:
        # Test that all imports work
        import sys
        import os
        # Add other imports from the code here
    except ImportError as e:
        failures.append(f"Import Error: {{e}}")
    
    try:
        # Test that main classes/functions can be instantiated/called without errors
        # Example: game = GameClass()  # Don't actually run the game loop
    except Exception as e:
        failures.append(f"Initialization Error: {{e}}")
    
    if failures:
        print("Error executing code:")
        for f in failures:
            print(f)
    else:
        print("All validation checks passed!")

run_tests()
```

## **OUTPUT FORMAT FOR SKIP TEST**:
```python
pass
```

# **Guidelines**
- Do not repeat original code in tests
- Output should contain only run_tests() function without explanations
- For complex applications, focus on validation rather than functionality testing
- Ensure tests don't trigger GUI windows or interactive elements
'''

TEXT_VALIDATION_PROMPT = f'''
You are a subtask result evaluator responsible for determining whether a subtask result meets the subtask requirements, if not, you need to improve it.

# Objective and Steps  
1. **Completeness and Quality Check:**  
   - Verify that the result includes all required elements of the task.  
   - Evaluate whether the output meets overall quality criteria (accuracy, clarity, formatting, and completeness).  

2. **Change Detection:**  
   - If this is a subsequent result, compare it with previous iterations.  
   - If the differences are minimal or the result has not significantly improved, consider it "good enough" for finalization.  

3. **Feedback and Escalation:**  
   - If the result meets the criteria or the improvements are negligible compared to previous iterations, return **"OK"**.  
   - Otherwise, provide **direct and precise feedback** and **output the improved result in the required format** for finalization.  

4. **Ensure Completeness:**
   - Your output must meet all requirements of the subtask.
   - Include all necessary details so that the output is self-contained and can be directly used as input for downstream tasks.


# Response Format  
- **If the result meets the standard:**  
  - Return **"OK"**.  

- **If the result does not meet the standard:**  
  - add detailed jusification for the change start with "here are some feedbacks" and directly write an improved new result start with "here are the changes".
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
      "output_format": "JSON",
      "data": "Aggregated customer feedback data ready for analysis."
    },
    "task1": {
      "objective": "Analyze the sentiment of collected feedback.",
      "agent_id": 1,
      "next": [],
      "prev": ["task0"],
      "status": "failed",
      "output_format": "Markdown report",
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
       - Consider improve the whole workflow by modifying, deleting or adding tasks.
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
You are a workflow result synthesizer responsible for extracting, connecting, and integrating results from all completed subtasks to produce the final deliverable that fully addresses the original task requirements.

# Input Format
[TASK]: The original task description and requirements
[CHATHISTORY]: Complete workflow execution results containing all subtask outputs, organized by task order

# Core Objective
Extract and synthesize results from ALL subtasks in the workflow to create a comprehensive final solution. This is NOT a summary - it's the actual deliverable that the user requested.

# Critical Instructions

1. **Extract All Relevant Results**: 
   - Identify completed subtasks and their outputs from [CHATHISTORY]
   - Extract key deliverables, insights, code, analysis, or content from each subtask
   - Ensure no important results are overlooked or omitted

2. **Connect and Integrate Outputs**:
   - Logically connect outputs from different subtasks
   - Resolve dependencies between subtask results
   - Combine partial solutions into a complete whole
   - Build upon earlier subtask results in later sections

3. **Respect Original Task Format Requirements**:
   - If [TASK] explicitly requests code (e.g., "Write a Python script"), provide complete executable code
   - If [TASK] requests documents (e.g., "Create a report"), provide the full document content
   - If [TASK] requests analysis, provide comprehensive findings with supporting evidence
   - If no specific format is mentioned, use clear, well-structured text or markdown

4. **Ensure Completeness and Quality**:
   - The output must fully satisfy ALL requirements mentioned in [TASK]
   - Include all necessary components, not just the final step
   - Provide a standalone deliverable that requires no additional work
   - Maintain professional quality and coherence throughout

5. **Output Structure Guidelines**:
   - Start directly with the deliverable content (no meta-commentary)
   - For code tasks: Provide complete, executable code with necessary imports and functions
   - For document tasks: Include proper headings, sections, and formatting
   - For analysis tasks: Present findings, methodology, and conclusions clearly
   - For creative tasks: Deliver the requested content in full

# What NOT to Do
- Don't provide just a summary or overview of what was done
- Don't include process descriptions or workflow commentary  
- Don't reference "subtasks" or "workflow steps" in the final output
- Don't provide incomplete or partial solutions
- Don't add meta-commentary about the extraction process

# Success Criteria
The output should be indistinguishable from a high-quality deliverable created directly for the original task, incorporating all insights and results discovered during the workflow execution.
'''