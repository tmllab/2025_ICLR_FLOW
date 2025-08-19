#!/usr/bin/env python3
"""
测试修复后的execute逻辑
"""
import asyncio
from unittest.mock import Mock, AsyncMock
from runner import AsyncRunner
from workflow import Workflow, Task
from history import History

class TestValidator:
    def __init__(self, fail_attempts=0):
        self.call_count = 0
        self.fail_attempts = fail_attempts
    
    async def validate(self, task_objective, result, history):
        self.call_count += 1
        print(f"    Validation call {self.call_count}: {task_objective[:30]}...")
        
        if self.call_count <= self.fail_attempts:
            return f"Feedback for attempt {self.call_count}", "failed"
        else:
            return "Task validated successfully", "completed"

class TestTaskExecuter:
    def __init__(self):
        self.execute_calls = 0
        self.re_execute_calls = 0
    
    async def execute(self, task_objective, agent_id, context, next_objective):
        self.execute_calls += 1
        print(f"    Initial execute: attempt {self.execute_calls}")
        return f"Execute result {self.execute_calls}"
    
    async def re_execute(self, task_objective, context, next_objective, previous_result, history):
        self.re_execute_calls += 1
        print(f"    Re-execute: attempt {self.re_execute_calls}")
        return f"Re-execute result {self.re_execute_calls}"

class TestTask:
    def __init__(self):
        self.objective = "Test task objective"
        self.status = "pending"
        self.history_saves = []
        self.status_changes = []
    
    def save_history(self, result, feedback):
        self.history_saves.append((result, feedback))
        print(f"    Saved history: result='{result[:30]}...', feedback='{feedback[:30]}...'")
    
    def set_status(self, status):
        self.status = status
        self.status_changes.append(status)
        print(f"    Status set to: {status}")
    
    def get_history(self):
        return [{"role": "user", "content": f"Previous attempt {i+1}"} for i in range(len(self.history_saves))]

async def test_execute_logic():
    print("Testing execute logic...")
    
    # Test 1: 立即通过验证
    print("\n=== Test 1: Pass on first validation ===")
    runner = AsyncRunner("test", max_validation_itt=3)
    runner.validator = TestValidator(fail_attempts=0)
    runner.executer = TestTaskExecuter()
    
    task = TestTask()
    workflow = Mock()
    workflow.get_context.return_value = "Test context"
    workflow.get_downsteam_objectives.return_value = "Test downstream"
    
    result = await runner.execute(workflow, "task1")
    print(f"  Final result: {result}")
    print(f"  Final status: {task.status}")
    print(f"  History saves: {len(task.history_saves)}")
    print(f"  Execute calls: {runner.executer.execute_calls}")
    print(f"  Re-execute calls: {runner.executer.re_execute_calls}")
    
    # Test 2: 第2次通过验证
    print("\n=== Test 2: Pass on second validation ===")
    runner2 = AsyncRunner("test", max_validation_itt=3)
    runner2.validator = TestValidator(fail_attempts=1)
    runner2.executer = TestTaskExecuter()
    
    task2 = TestTask()
    
    result2 = await runner2.execute(workflow, "task2")
    print(f"  Final result: {result2}")
    print(f"  Final status: {task2.status}")
    print(f"  History saves: {len(task2.history_saves)}")
    print(f"  Execute calls: {runner2.executer.execute_calls}")
    print(f"  Re-execute calls: {runner2.executer.re_execute_calls}")
    
    # Test 3: 永远不通过验证
    print("\n=== Test 3: Never pass validation ===")
    runner3 = AsyncRunner("test", max_validation_itt=3)
    runner3.validator = TestValidator(fail_attempts=10)  # 永远失败
    runner3.executer = TestTaskExecuter()
    
    task3 = TestTask()
    
    result3 = await runner3.execute(workflow, "task3")
    print(f"  Final result: {result3}")
    print(f"  Final status: {task3.status}")
    print(f"  History saves: {len(task3.history_saves)}")
    print(f"  Execute calls: {runner3.executer.execute_calls}")
    print(f"  Re-execute calls: {runner3.executer.re_execute_calls}")

# 模拟workflow.tasks访问
class MockWorkflow:
    def __init__(self, tasks):
        self.tasks = tasks
    
    def get_context(self, task_id):
        return "Mock context"
    
    def get_downsteam_objectives(self, task_id):
        return "Mock downstream objectives"

async def test_with_mock_workflow():
    """使用真实的workflow对象进行测试"""
    print("Testing execute logic with mock workflow...")
    
    # Test 1: 立即通过验证
    print("\n=== Test 1: Pass on first validation ===")
    runner = AsyncRunner("test", max_validation_itt=3)
    runner.validator = TestValidator(fail_attempts=0)
    runner.executer = TestTaskExecuter()
    
    # 创建真实的Task对象
    task = Task("task1", "Test task objective", 1, [], [], "pending", History(), "Test Agent")
    workflow = MockWorkflow({"task1": task})
    
    result = await runner.execute(workflow, "task1")
    print(f"  Final result: {result}")
    print(f"  Final status: {task.status}")
    print(f"  Execute calls: {runner.executer.execute_calls}")
    print(f"  Re-execute calls: {runner.executer.re_execute_calls}")
    
    # Test 2: 第2次通过验证  
    print("\n=== Test 2: Pass on second validation ===")
    runner2 = AsyncRunner("test", max_validation_itt=3)
    runner2.validator = TestValidator(fail_attempts=1)
    runner2.executer = TestTaskExecuter()
    
    task2 = Task("task2", "Test task objective", 1, [], [], "pending", History(), "Test Agent")
    workflow2 = MockWorkflow({"task2": task2})
    
    result2 = await runner2.execute(workflow2, "task2")
    print(f"  Final result: {result2}")
    print(f"  Final status: {task2.status}")
    print(f"  Execute calls: {runner2.executer.execute_calls}")
    print(f"  Re-execute calls: {runner2.executer.re_execute_calls}")
    
    # Test 3: 永远不通过验证
    print("\n=== Test 3: Never pass validation ===")
    runner3 = AsyncRunner("test", max_validation_itt=3)
    runner3.validator = TestValidator(fail_attempts=10)  # 永远失败
    runner3.executer = TestTaskExecuter()
    
    task3 = Task("task3", "Test task objective", 1, [], [], "pending", History(), "Test Agent")
    workflow3 = MockWorkflow({"task3": task3})
    
    result3 = await runner3.execute(workflow3, "task3")
    print(f"  Final result: {result3}")
    print(f"  Final status: {task3.status}")
    print(f"  Execute calls: {runner3.executer.execute_calls}")
    print(f"  Re-execute calls: {runner3.executer.re_execute_calls}")

if __name__ == "__main__":
    asyncio.run(test_with_mock_workflow())