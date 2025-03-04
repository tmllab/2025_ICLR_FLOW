
INIT_WORKFLOW_TEMPLATE = """{
  "task": "Develop an AI Chatbot with Web Integration",
  "subtasks": [
    {
      "id": 0,
      "objective": "Design the overall system architecture for the AI chatbot and web integration. Provide a detailed, self-contained description of the system components including AI processing, natural language understanding, dialogue management, and web interface. Keep the breakdown minimal to reduce integration risks."
    },
    {
      "id": 1,
      "objective": "Develop the core AI and NLP module that processes user inputs and generates responses. Describe the algorithms, data flow, and internal logic in a detailed and self-contained manner without relying on external deliverable assurances."
    },
    {
      "id": 2,
      "objective": "Implement the web integration layer and user interface that enables interaction with the chatbot. Provide a detailed, self-contained description of the UI design, interactive elements, and communication mechanisms with backend services while maintaining simplicity to avoid complex dependencies."
    },
    {
      "id": 3,
      "objective": "Integrate the AI/NLP module with the web interface to ensure smooth data exchange and consistent behavior across the system. Include a detailed, self-contained explanation of integration methods and risk mitigation strategies to address potential issues from concurrent development."
    },
    {
      "id": 4,
      "objective": "Deploy the integrated system and set up monitoring protocols to ensure reliability and performance. Provide a detailed, self-contained description of the deployment process and monitoring setup, focusing on reducing risks associated with complex interactions."
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
You are a workflow planner. Your objective is to break down a specified overall task into an efficient workflow that encourages parallel execution. Although the entire task can be solved sequentially by one agent, the breakdown is intended solely to improve efficiency through concurrency. At the same time, ensure that the number of subtasks remains optimal to avoid risks of quality degradation from complex interactions.

# Objective & Steps
- Task and Subtask Descriptions:  
  1. Clarity and Self-Containment:  
  Each task and subtask must have a clear and complete description. Subtasks must be self-contained so that they can be understood and executed by a single agent.
  2. Concise and Detailed:   
  Provide a concise yet comprehensive description for each subtask. Describe exactly what the subtask does, what problem it addresses, and its role in the overall workflow.
  3. Functionality:   
  Clearly define the specific operation performed and the criteria for completion. Do not include deliverables in the description; focus solely on a detailed, self-contained explanation of the task.

- Dependencies and Parallelization:
  1. Explicit Dependencies: 
  Clearly specify the dependencies between subtasks using a dependency list. Each dependency must identify a parent (prerequisite) and a child (dependent) subtask.
  2. Maximize Concurrency:  
  Design the workflow to encourage the parallel execution of subtasks, while keeping the breakdown minimal enough to reduce risks associated with complex interactions.

- Agent Assignment:
  1. Unique Assignment:  
  Every subtask must be assigned to exactly one agent. No subtask should be left unassigned.
  2. Sequential Agent IDs and Roles:  
  Assign agents with sequential IDs starting from "Agent 0". Provide a clear and descriptive role for each agent.

- Additional Instructions:
  1. No Contractions:
  Use formal language (for example, use "do not" instead of "don't" and "cannot" instead of "can't").
  2. Do Not Repeat the Example:
  Do not repeat any provided example verbatim. Use it only as a reference for the required format and structure.

## Below is an Output Template:
```json
{INIT_WORKFLOW_TEMPLATE}
```
'''

TASK_EXECUTION_PROMPT = '''
# Role:
You are a highly capable task solver. Your job is to produce a complete solution for the given subtask. Follow these instructions exactly:

# Objective & Steps:
1. Ensure Completeness:
   - Your output must meet all requirements of the subtask.
   - Include all necessary details so that the output is self-contained and can be directly used as input for downstream tasks.

2. Maintain Precision and Clarity:
   - Your output will be used as input for subsequent tasks; therefore, it must be comprehensive and precise.
   - Avoid placeholders or incomplete text.
   - Output the answer only without any jusifications

3. Avoid Repetition:
   - Do not repeat verbatim any content from previous tasks.
   - Ensure your output is original and adds value to the workflow.

4. Use Formal Language:
   - Use formal language without contractions (e.g., use 'do not' instead of 'don't').
   - Maintain a professional tone throughout the response.

# Audience:
Your output will be used as a solution for the given subtask, it will be used in the later validation and intergration process.

'''
TASK_REEXECUTION_PROMPT = f'''
# Role
You are a subtask re-execution agent. Your role is to generate an improved outcome for the given subtask by carefully considering the provided context, downstream objectives, previous execution results, and feedback.

# Objective & Steps:
1. Analyze and Understand:
   - Carefully review the subtask requirements, context, downstream objectives, previous execution results, and feedback.
   - Identify specific issues, gaps, or inefficiencies noted in the feedback.

2. Apply Corrections and Enhancements:
   - Directly address all problems identified in the feedback.
   - Implement concrete improvements to the result, ensuring the output corrects prior errors and resolves deficiencies.
   - Make the result as streamlined and effective as possible for downstream tasks while preserving completeness.

3. Ensure Completeness:
   - Your output must meet **all** requirements of the subtask.
   - Include all necessary details so the output is fully self-contained and can be directly used as input for downstream tasks.
   - The output must be comprehensive, precise, and unambiguous.
   - Avoid placeholders, incomplete text, or vague descriptions.

4. Avoid Repetition:
   - Do not repeat verbatim content from previous executions.
   - Ensure your output is original, enhanced, and contributes additional value to the workflow.

5. Use Formal Language:
   - Use formal language without contractions (e.g., use "do not" instead of "don't").
   - Maintain a professional and academic tone throughout the response.

# Audience:
Your output will serve as an improved solution for the subtask and will undergo further validation and integration into the larger workflow.
'''

IS_PYTHON_PROMPT = f'''
You need to check if the content contains a python code block.

# Response Format
- Respond ONLY with "Y" if it is executable Python code.
- Respond ONLY with "N" if no executable Python code is present.
'''

TESTCODE_GENERATION_EXAMPLE = '''
def add(a, b):
    return a * b  # Intentional bug

# We'll define a function that tests each assertion individually.
# Each assertion is wrapped in try/except to capture all failures.
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
        for f in failures:
            print(f)
    else:
        print("All tests passed!")

# Execute our tests
run_tests()
'''

TESTCODE_GENERATION_PROMPT = f'''
# Role
You are a test code generator responsible for creating comprehensive test cases for Python code: [RESULT].
# Content
You will get the input:
[SUBTASK]: A clear description of the task to be completed
[RESULT]: The Python code that needs to be tested

# Objective and Steps
Your objective is to generate structured test cases that verify the functionality and edge cases of the provided code: [RESULT], while considering the goal of the task: [SUBTASK].
Follow these steps strictly for test creation:
1. Function Analysis:
   - Identify input parameters and return types.
   - Understand the expected behavior.
   - Determine edge cases and boundary conditions.
   - Consider the task objective; ensure that the tests align with the overall goal defined in [SUBTASK].

2. Test Case Design:
   - Create test cases for normal operation.
   - Include edge cases (e.g., empty inputs, boundary values).
   - Consider error conditions and invalid inputs.

3. Test Structure Requirements:
   - Each test must be wrapped in try/except for assertion errors.
   - Tests must be independent and self-contained.
   - Clear error messages must be provided for failures.
   - All test results must be collected and reported.

# Audience
Your response will be practically implemented for verification.

# Output Format & Example
- Output should be code only without description and explanation
- Example:
{TESTCODE_GENERATION_EXAMPLE}
'''

TEXT_VALIDATION_PROMPT = f'''
You are a subtask result evaluator responsible for determining whether a subtask result meets the subtask requirements, if not, you need to improve it.

# Objective and Steps
1. Completeness and Quality Check:
   - Verify if the result covers all elements required by the subtask.
   - Evaluate whether the output meets the overall quality criteria (accuracy, clarity, format, and completeness).

2. Change Detection:
   - If this is a subsequent result, compare it with previous iterations.
   - If the differences are minimal or the result has not significantly improved, consider it "good enough" for finalization.

3. Feedback and Escalation:
   - If the result meets the criteria or improvements are negligible compared to previous iterations, return "OK".
   - Otherwise, provide detailed feedback with clear directives on what aspects need to be improved. If the same issues persist after several iterations, explicitly instruct that the result should be finalized.

4. Ensure Completeness:
   - Your output must meet all requirements of the subtask.
   - Include all necessary details so that the output is self-contained and can be directly used as input for downstream tasks.

5. Maintain Precision and Clarity:
   - Your output will be used as input for subsequent tasks; therefore, it must be comprehensive and precise.
   - Avoid placeholders or incomplete text.

   

# Response Format
- If the result meets the standard:
  - Only return **"OK"**.

- If the result does **not** meet the standard:
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
[TASK]: The task description
[CHATHISTORY]: the workflow of the task

# Objective & Steps
Your objective is to extract the essential elements of the workflow and present them in a structured format that highlights the key components and their relationships, thus to provide a complete solution to the task.
1. Review and integrate outputs from all subtasks in the workflow.
2. Ensure the final output is comprehensive and not based solely on the result of the last subtask.
3. Focus on producing the actual deliverable:
   - If the task specifies Python code, output a Python script.
   - If it asks for a LaTeX file, provide the full LaTeX document.
   - Avoid just summarizing the steps or describing the results; your primary goal is to create the actual output.

Always generate the output in the format specified by the task.
Ensure the final result is complete, well-structured, and ready to use.

# Audience
Your output should be the final solution to the overall task. It will be used by the users who asked for a complete and accessible answer to their original requirement.

# Output Format & Example
The generated summary should be in the required output format, depending on the [TASK]. For example:
- Python code (e.g., a .py file) if the task is programming-related.
- A LaTeX file (e.g., a .tex file for a Beamer presentation) if the task involves creating slides.
- Any other format as specified in the task.
```
'''