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

    async def validate(self, task_obj, result, history):
        print('------Run textValidator.validate()------')

        user_content = f"""
        ## current task Requirement:
        {task_obj}

        ---

        ## current task change History:
        {history}

        ---

        ## current task Latest Result:
        {result}
        """

        messages = [
            {'role': 'system', 'content': self.text_validation_prompt},
            {'role': 'user', 'content': user_content}
        ]
        print(user_content)

        feedback = await self.gpt_client.a_chat_completion(messages, temperature=Config.TEMPERATURE)
        with open('validate_log.json', 'a', encoding='utf-8') as file:
            file.write('----------\nGOT TEXT VALIDATION\n----------')
            json.dump({'task_obj': task_obj, 'result': result, 'feedback': feedback}, file, indent=4)

        
        if "OK" in feedback and len(feedback)<50:
            return None, 'completed'
        else:
            return feedback, 'failed'
    
