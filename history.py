class History:
    def __init__(self):
        self.data = []

    def store(self, result: str, feedback: str = ''):
        self.data.append({'result': result, 'feedback': feedback})

    def get_history(self, idx=None):
        if not self.data:
            return None

        if idx is None:
            return self.data
        else:
            if isinstance(idx, int) and 0 <= idx < len(self.data):
                return self.data[idx]
            else:
                return None
        
    def get_latest(self):
        if not self.data:
            return None
        return self.data[-1]

    def __str__(self):
        return f"History={self.data}"