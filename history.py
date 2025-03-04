class History:
    def __init__(self):
        self.data = []

    def save(self, result: str, feedback: str = None):
        entry = {'result': result}
        if feedback:
            entry['feedback'] = feedback
        self.data.append(entry)

    def get_history_by_index(self, index: int) -> tuple[str, str]:
        if len(self.data)==0:
            return "", ""
        if not (-len(self.data) <= index < len(self.data)):
            raise IndexError(f"Invalid index: {index}. History has {len(self.data)} entries.")
        d = self.data[index]
        result = d.get('result', 'No result')
        feedback = d.get('feedback', 'No feedback')
        return result, feedback

    def get_result_by_index(self, index: int) -> str:    
        result, _ = self.get_history_by_index(index)
        return result

    def get_feedback_by_index(self, index: int) -> str:
        _, fb = self.get_history_by_index(index)
        return fb
    
    def get_latest_history(self):
        result, fb = self.get_history_by_index(-1)
        return result, fb
    
    def get_latest_result(self):
        result, _ = self.get_history_by_index(-1)
        return result        
    
    def get_latest_feedback(self):
        _, fb = self.get_history_by_index(-1)
        return fb

    def get_history(self):
        if len(self.data)==0:
            return ""
        history = []
        for i in range(len(self.data)):
            result, feedback = self.get_history_by_index(i)
            history.append(
                f"{'-'*40}\n"
                f"revision {i}:\n"
                f"Result:\n{result}\n\n"
                f"Feedback:\n{feedback}\n"
            )

        history.append('-' * 40)
        return "\n".join(history)
            


    def __str__(self):
        return self.get_history()
    

    def to_dict(self) -> dict:
        return {
            'history': [
                {
                    'revision': i,
                    'result': entry.get('result', 'No result'),
                    'feedback': entry.get('feedback')
                }
                for i, entry in enumerate(self.data)
            ]
        }