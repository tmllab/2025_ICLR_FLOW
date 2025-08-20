from gptClient import GPTClient
from config import Config
import json
import prompt
from logging_config import get_logger, log_validation_result

class textValidator:
    """
    A validator class that checks text-based task results using GPT.

    This class is responsible for:
    - Validating text-based task outputs
    - Providing feedback on task results
    - Determining if task requirements are met
    - Logging validation results

    Attributes:
        gpt_client (GPTClient): Client for interacting with GPT API.
        text_validation_prompt (str): Prompt template for text validation.
    """

    def __init__(self):
        """
        Initialize the textValidator with GPT client and validation prompt.

        Sets up:
        - GPT client with configuration parameters
        - Text validation prompt template
        """
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.text_validation_prompt = prompt.TEXT_VALIDATION_PROMPT
        self.logger = get_logger('validation')

    def _is_validation_successful(self, feedback: str) -> bool:
        """
        Determine if validation feedback indicates success based on the prompt design.
        
        According to TEXT_VALIDATION_PROMPT:
        - If result meets standard: Return "OK"
        - If result doesn't meet standard: Provide detailed feedback starting with 
          "here are some feedbacks" and improvements starting with "here are the changes"
        
        Args:
            feedback (str): The GPT validation response
            
        Returns:
            bool: True if validation is successful (feedback is "OK")
        """
        feedback_stripped = feedback.strip()
        
        # Exact match for "OK" as specified in the prompt
        if "OK" in feedback_stripped.upper():
            return True
            
  
            
        return False

    async def validate(self, task_obj, result, history, overall_task: str = None, output_format: str = None) -> tuple[str, str]:
        """
        Validate text-based task results using GPT with improved logic.

        This method:
        1. Constructs a validation prompt with task requirements, history, and result
        2. Uses GPT to evaluate the result
        3. Applies intelligent success/failure detection
        4. Logs validation results with proper error handling
        5. Returns structured feedback

        Args:
            task_obj: The task objective/requirement.
            result (str): The text result to validate.
            history: Previous execution history and feedback.

        Returns:
            tuple[str, str]: A tuple containing (feedback, status).
                           status can be 'completed' or 'failed'.
                           feedback contains validation results or improvement suggestions.
        """
        try:
            system_message = {
                "role": "system",
                "content": self.text_validation_prompt
            }

            # Build user message with overall context
            content_parts = []
            
            # Add overall task context if available
            if overall_task:
                content_parts.append(f"## Overall User Goal:\n{overall_task}\n\n---\n")
            
            content_parts.extend([
                f"## Current Task Requirement:\n{task_obj}\n\n---\n"
            ])
            
            # Add output format requirement if specified
            if output_format and output_format.strip():
                content_parts.append(f"## Required Output Format:\n{output_format}\n\n---\n")
            
            content_parts.append(f"## Current Task Latest Result:\n{result}")
            
            user_message = {
                "role": "user",
                "content": "".join(content_parts)
            }

            # Combine messages properly
            messages = [system_message]
            if history:
                messages.extend(history)
            messages.append(user_message)

            feedback = await self.gpt_client.a_chat_completion(messages)
            
            # Determine validation success using improved logic
            if self._is_validation_successful(feedback):
                status = 'completed'
                result_msg = "there are no additional problems"
                self.logger.info("Text validation passed")
            else:
                status = 'failed'
                result_msg = feedback
                self.logger.warning(f"Text validation failed: {feedback[:100]}...")
            
            # Log validation results using structured logging
            task_id = getattr(history, 'task_id', 'unknown') if history and hasattr(history, 'task_id') else 'unknown'
            log_validation_result(
                task_id=task_id,
                task_obj=task_obj,
                result=result,
                validation_type='text',
                status=status,
                feedback=feedback
            )
            
            return result_msg, status
                
        except Exception as e:
            return f"Validation error: {str(e)}", 'failed'
    
