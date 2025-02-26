from gptClient import GPTClient
from config import Config
import json
import prompt

class textValidator:
    def __init__(self):
        self.gpt_client = GPTClient(
            api_key=Config.OPENAI_API_KEY,
            model=Config.GPT_MODEL,
            temperature=Config.TEMPERATURE
        )
        self.text_validation_prompt = prompt.TEXT_VALIDATION_PROMPT

    async def validate(self, task_obj, result):
        print('------Run textValidator.validate()------')

        user_content = f'''
            Here is the subtask: {task_obj}
            Here is the result: {result}
        '''

        messages = [
            {'role': 'system', 'content': self.text_validation_prompt},
            {'role': 'user', 'content': user_content}
        ]

        feedback = await self.gpt_client.a_chat_completion(messages, temperature=Config.TEMPERATURE)
        with open('validate_log.json', 'a', encoding='utf-8') as file:
            file.write('----------\nGOT TEXT VALIDATION\n----------')
            json.dump({'task_obj': task_obj, 'result': result, 'feedback': feedback}, file, indent=4)

        
        if feedback == "NONE":
            return None
        else:
            return feedback
    
