class History:
    def __init__(self):
        self.data = []

    def save(self, result: str, feedback: str = ''):
        self.data.append({'result': result, 'feedback': feedback})

    def get_history(self):
        return str(self.data)

    def get_history_index(self, idx = None):
        if not self.data:
            return None

        if idx is None:
            return self.data
        else:
            if isinstance(idx, int) and 0 <= idx < len(self.data):
                return self.data[idx]['result'], self.data[idx]['feedback']
            else:
                return None
        
    def get_latest_history(self):
        if not self.data:
            return None
        return self.data[-1]['result'], self.data[-1]['feedback']

    def __str__(self):
        return f"History= {self.data}"
    
    def __dict__(self):
        return {'history': self.data}
        