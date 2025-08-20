import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import atexit

class FlowLogger:
    """
    Centralized logging system for the Flow framework.
    
    Features:
    - Unified logging configuration across all modules
    - Structured JSON logging for results
    - Automatic log rotation
    - Thread-safe logging
    - Multiple log levels and formatters
    - Separate log files for different components
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for logging instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        
        # Create a unique run ID and folder for this execution
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path("runs") / f"run_{self.run_id}"
        
        # Create run-specific subdirectories
        self.log_dir = self.run_dir / "logs"
        self.results_dir = self.run_dir / "results"
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Log files
        self.main_log = self.log_dir / "flow_main.log"
        self.error_log = self.log_dir / "flow_errors.log"
        self.validation_log = self.log_dir / "validation.log"
        self.workflow_log = self.log_dir / "workflow.log"
        self.execution_log = self.log_dir / "execution.log"
        
        # Initialize loggers
        self._setup_loggers()
        
        # Register cleanup
        atexit.register(self.cleanup)
    
    def _setup_loggers(self):
        """Configure all loggers with appropriate handlers and formatters."""
        
        # Main logger configuration
        self.main_logger = logging.getLogger('flow.main')
        self.main_logger.setLevel(logging.INFO)
        
        # Error logger
        self.error_logger = logging.getLogger('flow.error')
        self.error_logger.setLevel(logging.ERROR)
        
        # Component-specific loggers
        self.validation_logger = logging.getLogger('flow.validation')
        self.validation_logger.setLevel(logging.DEBUG)
        
        self.workflow_logger = logging.getLogger('flow.workflow')
        self.workflow_logger.setLevel(logging.INFO)
        
        self.execution_logger = logging.getLogger('flow.execution')
        self.execution_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        for logger in [self.main_logger, self.error_logger, self.validation_logger, 
                      self.workflow_logger, self.execution_logger]:
            logger.handlers.clear()
            logger.propagate = False
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        json_formatter = JsonFormatter()
        
        # Console handler (for main logger)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.main_logger.addHandler(console_handler)
        
        # File handlers with rotation
        main_file_handler = logging.handlers.RotatingFileHandler(
            self.main_log, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        main_file_handler.setLevel(logging.DEBUG)
        main_file_handler.setFormatter(detailed_formatter)
        self.main_logger.addHandler(main_file_handler)
        
        # Error file handler
        error_file_handler = logging.handlers.RotatingFileHandler(
            self.error_log, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(detailed_formatter)
        self.error_logger.addHandler(error_file_handler)
        
        # Component-specific file handlers
        validation_handler = logging.handlers.RotatingFileHandler(
            self.validation_log, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        validation_handler.setLevel(logging.DEBUG)
        validation_handler.setFormatter(json_formatter)
        self.validation_logger.addHandler(validation_handler)
        
        workflow_handler = logging.handlers.RotatingFileHandler(
            self.workflow_log, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        workflow_handler.setLevel(logging.INFO)
        workflow_handler.setFormatter(json_formatter)
        self.workflow_logger.addHandler(workflow_handler)
        
        execution_handler = logging.handlers.RotatingFileHandler(
            self.execution_log, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        execution_handler.setLevel(logging.INFO)
        execution_handler.setFormatter(json_formatter)
        self.execution_logger.addHandler(execution_handler)
    
    def get_logger(self, component: str = 'main') -> logging.Logger:
        """Get a logger for a specific component."""
        loggers = {
            'main': self.main_logger,
            'error': self.error_logger,
            'validation': self.validation_logger,
            'workflow': self.workflow_logger,
            'execution': self.execution_logger
        }
        return loggers.get(component, self.main_logger)
    
    def log_workflow_event(self, event_type: str, data: Dict[str, Any], task_id: Optional[str] = None):
        """Log structured workflow events."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'task_id': task_id,
            'data': data
        }
        self.workflow_logger.info(json.dumps(log_entry, ensure_ascii=False, default=str))
    
    def log_validation_result(self, task_id: str, task_obj: str, result: str, 
                            validation_type: str, status: str, feedback: str):
        """Log validation results in structured format."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'task_id': task_id,
            'task_objective': task_obj[:200] + '...' if len(task_obj) > 200 else task_obj,
            'result': result[:500] + '...' if len(result) > 500 else result,
            'validation_type': validation_type,
            'status': status,
            'feedback': feedback
        }
        self.validation_logger.info(json.dumps(log_entry, ensure_ascii=False, default=str))
    
    def log_execution_result(self, task_id: str, agent_id: str, objective: str, 
                           result: str, status: str, duration: float):
        """Log task execution results."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'task_id': task_id,
            'agent_id': agent_id,
            'objective': objective,
            'result': result[:1000] + '...' if len(result) > 1000 else result,
            'status': status,
            'duration_seconds': duration
        }
        self.execution_logger.info(json.dumps(log_entry, ensure_ascii=False, default=str))
    
    def save_results(self, filename: str, data: Dict[str, Any], description: str = ""):
        """Save results to JSON files with metadata."""
        filepath = self.results_dir / filename
        
        # Add metadata
        result_data = {
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'data': data
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
            self.main_logger.info(f"Results saved to {filepath}")
        except Exception as e:
            self.error_logger.error(f"Failed to save results to {filepath}: {e}")
    
    def log_workflow_summary(self, overall_task: str, workflow_data: Dict[str, Any], 
                           duration: float, summary_result: str):
        """Log complete workflow execution summary."""
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_task': overall_task,
            'total_tasks': len(workflow_data.get('tasks', {})),
            'completed_tasks': len([t for t in workflow_data.get('tasks', {}).values() 
                                  if t.get('status') == 'completed']),
            'failed_tasks': len([t for t in workflow_data.get('tasks', {}).values() 
                               if t.get('status') == 'failed']),
            'total_duration_seconds': duration,
            'agents_used': len(workflow_data.get('agents', [])),
            'summary_result_length': len(summary_result)
        }
        
        self.workflow_logger.info(json.dumps(summary_data, ensure_ascii=False))
        
        # Save complete results
        self.save_results(
            f"workflow_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            {
                'task': overall_task,
                'workflow': workflow_data,
                'summary': summary_result,
                'execution_summary': summary_data
            },
            f"Complete workflow execution for: {overall_task[:50]}..."
        )
    
    def get_run_directory(self) -> Path:
        """Get the current run directory path."""
        return self.run_dir
    
    def get_run_id(self) -> str:
        """Get the current run ID."""
        return self.run_id
    
    def copy_workflow_file(self, source_file: str, destination_name: str = None):
        """Copy workflow-related files (like initflow.json) into the run directory."""
        import shutil
        source_path = Path(source_file)
        if source_path.exists():
            dest_name = destination_name or source_path.name
            dest_path = self.run_dir / dest_name
            shutil.copy2(source_path, dest_path)
            self.main_logger.info(f"Copied {source_file} to run directory as {dest_name}")
        else:
            self.main_logger.warning(f"Source file {source_file} not found, skipping copy")
    
    def save_run_metadata(self, overall_task: str, config: dict):
        """Save run metadata including task description and configuration."""
        metadata = {
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
            "overall_task": overall_task,
            "configuration": config
        }
        
        metadata_file = self.run_dir / "run_metadata.json"
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            self.main_logger.info(f"Run metadata saved to {metadata_file}")
        except Exception as e:
            self.error_logger.error(f"Failed to save run metadata: {e}")
    
    def cleanup(self):
        """Clean up logging resources."""
        for logger in [self.main_logger, self.error_logger, self.validation_logger, 
                      self.workflow_logger, self.execution_logger]:
            for handler in logger.handlers:
                handler.close()

class JsonFormatter(logging.Formatter):
    """Custom formatter that outputs logs in JSON format."""
    
    def format(self, record):
        log_entry = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage()
        }
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)

# Global logger instance
flow_logger = FlowLogger()

def get_logger(component: str = 'main') -> logging.Logger:
    """
    Get a logger for the specified component.
    
    Args:
        component: Component name ('main', 'error', 'validation', 'workflow', 'execution')
        
    Returns:
        Configured logger instance
    """
    return flow_logger.get_logger(component)

def log_workflow_event(event_type: str, data: Dict[str, Any], task_id: Optional[str] = None):
    """Log a workflow event."""
    flow_logger.log_workflow_event(event_type, data, task_id)

def log_validation_result(task_id: str, task_obj: str, result: str, 
                         validation_type: str, status: str, feedback: str):
    """Log a validation result."""
    flow_logger.log_validation_result(task_id, task_obj, result, validation_type, status, feedback)

def log_execution_result(task_id: str, agent_id: str, objective: str, 
                        result: str, status: str, duration: float):
    """Log a task execution result."""
    flow_logger.log_execution_result(task_id, agent_id, objective, result, status, duration)

def save_results(filename: str, data: Dict[str, Any], description: str = ""):
    """Save results to a JSON file."""
    flow_logger.save_results(filename, data, description)

def log_workflow_summary(overall_task: str, workflow_data: Dict[str, Any], 
                        duration: float, summary_result: str):
    """Log complete workflow execution summary."""
    flow_logger.log_workflow_summary(overall_task, workflow_data, duration, summary_result)

def get_run_directory():
    """Get the current run directory."""
    return flow_logger.get_run_directory()

def get_run_id():
    """Get the current run ID."""
    return flow_logger.get_run_id()

def copy_workflow_file(source_file: str, destination_name: str = None):
    """Copy a workflow file to the run directory."""
    flow_logger.copy_workflow_file(source_file, destination_name)

def save_run_metadata(overall_task: str, config: dict):
    """Save run metadata."""
    flow_logger.save_run_metadata(overall_task, config)

# NEW: Intermediate Results Tracking Functions
def get_results_dir():
    """Get the current run's results directory."""
    return flow_logger.results_dir

def log_intermediate_result(task_id: str, iteration: int, result_type: str, 
                          data: Dict[str, Any], status: str = "in_progress"):
    """Log intermediate results during task execution, validation, and workflow refinement"""
    execution_logger = get_logger('execution')
    
    intermediate_data = {
        "event_type": "intermediate_result",
        "task_id": task_id,
        "iteration": iteration,
        "result_type": result_type,  # e.g., 'task_execution', 'validation_attempt', 'workflow_refinement'
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    
    execution_logger.info(json.dumps(intermediate_data, ensure_ascii=False, indent=2))

def log_gpt_conversation(task_id: str, conversation_type: str, messages: list, 
                       response: str, iteration: int = 1):
    """Log complete GPT conversations for debugging and analysis"""
    validation_logger = get_logger('validation')
    
    conversation_data = {
        "event_type": "gpt_conversation",
        "task_id": task_id,
        "conversation_type": conversation_type,  # e.g., 'task_execution', 'validation', 're_execution'
        "iteration": iteration,
        "messages": messages,
        "gpt_response": response,
        "timestamp": datetime.now().isoformat()
    }
    
    validation_logger.info(json.dumps(conversation_data, ensure_ascii=False, indent=2))

def save_intermediate_snapshot(file_name: str, data: Dict[str, Any], description: str, 
                             task_id: str = None, iteration: int = None):
    """Save intermediate state snapshots to results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create more specific filename with iteration info
    if task_id and iteration is not None:
        file_name = f"{timestamp}_{task_id}_iter{iteration}_{file_name}"
    else:
        file_name = f"{timestamp}_{file_name}"
        
    file_path = get_results_dir() / file_name
    
    snapshot_data = {
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "task_id": task_id,
        "iteration": iteration,
        "data": data
    }
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, ensure_ascii=False, indent=2)
        
        main_logger = get_logger('main')
        main_logger.info(f"Intermediate snapshot saved: {file_name}")
        
    except Exception as e:
        error_logger = get_logger('error')
        error_logger.error(f"Failed to save intermediate snapshot {file_name}: {e}")