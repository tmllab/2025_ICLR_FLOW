class History:
    def __init__(self):
        self.data = [{"results":"...", "feedback":"..."},{"results":"...", "feedback":"..."},{"results":"...", "feedback":"..."}]

    def to_str(self):
        pass 

    def get_history(self, idx=None):

        if len(self.data) == 0:
            return ""
        if idx==None:
            ##TODO output all history, e.g., return self_to_str()
            pass
        return self.data[idx]