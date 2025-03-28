﻿# [Flow: Modularized Agentic Workflow Automation (ICLR'25)](https://arxiv.org/abs/2501.07834)

## Note

Auto validation features (e.g., adding unit tests) has been added to main branch.

## Overview

**Flow** is a multi-agent framework designed to enhance task automation and execution efficiency through **modular workflow design** and **dynamic refinement**. By structuring workflows as **Activity-on-Vertex (AOV) graphs**, Flow enables real-time adjustments, supports concurrent execution of subtasks, and optimizes workflow structures dynamically.

An upcoming version of Flow will support tool usage (e.g., code terminal, web browser) and swarm workflow generation. This update will be integrated with **AG2** ([GitHub Repository](https://github.com/ag2ai/ag2)).

## Key Features

- **Modular Workflow Design**: Workflows are structured as **Activity-on-Vertex (AOV) graphs**, enabling **parallel execution** while minimizing dependencies.
- **Concurrency**: Subtasks execute concurrently based on predefined dependencies.
- **Dynamic Workflow Refinement**: The framework dynamically adapts workflows based on real-time performance feedback.
- **Error Handling & Recovery**: Flow prevents system-wide failures and subtask failures by auto validating each subtasks and modifying workflows in response to errors.

## How It Works

### 1. **Workflow Initialization**

Flow structures tasks as an **AOV graph**, where:

- Nodes represent individual subtasks.
- Directed edges define dependencies between subtasks.
- The system generates multiple workflow candidates and selects the most efficient structure based on **parallelism** and **dependency complexity**.

### 2. **Execution & Dynamic Updates**

During execution:

- Agents process subtasks concurrently according to their roles and dependencies.
- Performance feedback informs real-time workflow updates.
- New tasks may be introduced, reallocated, or modified to enhance execution efficiency.

### 3. **Workflow Refinement**

- Workflows are continuously refined during runtime.
- Failed tasks are dynamically reassigned or redesigned to prevent execution bottlenecks.
- The `refine_threshold` parameter determines when workflow refinement occurs.
- A lazy update strategy defers refinement until all active tasks are completed.

### 4. **Subtask Validation**

- Each time a task is completed, validation will be performed for `max_validation_itt` iterations.
- The validator will first determine the nature of the subtask. If it is Python code, the Python validator will be used; otherwise, the text validator will be employed.
- The Python validator will generate test code based on the subtask and result, and then execute it. The text validator will also generate corresponding feedback based on the subtask and result.
- The re-execute agent will regenerate a new result based on the previous outcome and feedback.

## Installation & Usage

### **Requirements**

- Python 3.8+
- OpenAI API

### **Installation**

```bash
# Clone the repository
git clone https://github.com/tmllab/2025_ICLR_FLOW.git
cd 2025_ICLR_FLOW
pip install -r requirements.txt
```

### **Configuration**

#### **Set API Key**

Linux/MacOS:

```bash
export OPENAI_API_KEY="your-api-key"
```

Windows (CMD):

```bash
set OPENAI_API_KEY="your-api-key"
```

#### **Set OpenAI Model**

```python
GPT_MODEL: str = "your-model"
```

#### **Define Your Task**

```python
overall_task: str = "your-task"
```

#### **Set Number of Candidate Graphs**

Configure the number of candidate workflow graphs.

```python
candidate_graphs: int = your-candidate-graphs
```

#### **Set Optimization Threshold**

Refinement will be triggered after completing the threshold amount of subtasks.  
Smaller values trigger more frequent updates.

```python
refine_threshold: int = your-threshold
```

#### **Set Max Optimization Iteration**

Configure the number of max optimization iteration times.

```python
max_refine_itt: int = your-optimization-iteration
```

#### **Set Max Validation Iteration**

Configure the number of max validation iteration times.
Note that the amount of this figure **can be 0** if you don't want any validation.

```python
max_validation_itt: int = your-validation-iteration
```

### **Run Flow**

```bash
python main.py
```

## **Examining Results**

Each subtask in the workflow includes:

- `id`: Unique subtask identifier.
- `objective`: Goal or content of the subtask.
- `agent_id`: Identifier of the agent executing the subtask.
- `next`: List of dependent subtasks.
- `prev`: List of prerequisite subtasks.
- `status`: Current execution status.
- `history`: Save the results and feedbacks of subtask.
- `remaining_dependencies`: Number of unfinished dependencies.
- `agent`: Assigned agent for execution.

### **Result Files**

- **`initflow.json`**: Stores the best-selected workflow among candidates during initialization.
- **`result.json`**: Stores the final workflow execution results, including outputs of all subtasks.
- **`example.txt`**: Contains the final synthesized output generated by the summary agent.

## Example Applications

To facilitate learning, we provide Jupyter notebooks with illustrative examples.

## Citation

If you find Flow useful, please consider citing our work:

```bibtex
@article{niu2025flow,
  title={Flow: Modularized Agentic Workflow Automation},
  author={Niu, Boye and Song, Yiliao and Lian, Kai and Shen, Yifan and Yao, Yu and Zhang, Kun and Liu, Tongliang},
  journal={ICLR},
  year={2025}
}
```
