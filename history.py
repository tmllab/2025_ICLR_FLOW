class History:
    """
    A class that manages execution history for tasks, storing results and feedback.

    This class is responsible for:
    - Storing execution results and feedback
    - Retrieving history entries by index
    - Providing various views of the history data
    - Converting history to different formats

    Attributes:
        data (list): List of dictionaries containing result and feedback entries.
    """

    def __init__(self):
        """
        Initialize an empty history container.
        """
        self.data = []

    def save(self, result: str, feedback: str = None):
        """
        Save a new history entry with result and optional feedback.

        Args:
            result (str): The execution result to save.
            feedback (str, optional): Associated feedback for the result. Defaults to None.
        """
        entry = {'result': result}
        if feedback:
            entry['feedback'] = feedback
        self.data.append(entry)

    def get_history_by_index(self, index: int) -> tuple[str, str]:
        """
        Retrieve both result and feedback for a specific history entry.

        Args:
            index (int): Index of the history entry to retrieve. Supports negative indexing.

        Returns:
            tuple[str, str]: A tuple containing (result, feedback) for the specified entry.

        Raises:
            IndexError: If the index is out of range for the history.
        """
        if len(self.data)==0:
            return "", ""
        if not (-len(self.data) <= index < len(self.data)):
            raise IndexError(f"Invalid index: {index}. History has {len(self.data)} entries.")
        d = self.data[index]
        result = d.get('result', 'No result')
        feedback = d.get('feedback', 'No feedback')
        return result, feedback

    def get_result_by_index(self, index: int) -> str:    
        """
        Get only the result from a specific history entry.

        Args:
            index (int): Index of the history entry.

        Returns:
            str: The result from the specified entry.
        """
        result, _ = self.get_history_by_index(index)
        return result

    def get_feedback_by_index(self, index: int) -> str:
        """
        Get only the feedback from a specific history entry.

        Args:
            index (int): Index of the history entry.

        Returns:
            str: The feedback from the specified entry.
        """
        _, fb = self.get_history_by_index(index)
        return fb
    
    def get_latest_history(self) -> tuple[str, str]:
        """
        Get the most recent history entry's result and feedback.

        Returns:
            tuple[str, str]: A tuple containing (result, feedback) from the latest entry.
        """
        result, fb = self.get_history_by_index(-1)
        return result, fb
    
    def get_latest_result(self) -> str:
        """
        Get only the result from the most recent history entry.

        Returns:
            str: The latest result.
        """
        result, _ = self.get_history_by_index(-1)
        return result        
    
    def get_latest_feedback(self) -> str:
        """
        Get only the feedback from the most recent history entry.

        Returns:
            str: The latest feedback.
        """
        _, fb = self.get_history_by_index(-1)
        return fb


    def get_history_as_chat_messages(self) -> list:
        """
        Get the history as a list of chat messages with separate roles:
        - 'assistant' for generated revisions
        - 'user' for feedback messages

        Returns:
            list: A list of dictionaries in chat format for OpenAI API
        """
        if not self.data:
            return []

        messages = []
        for i in range(len(self.data)):
            result, feedback = self.get_history_by_index(i)

            # Add the assistant's revision
            messages.append({
                "role": "assistant",
                "content": f"---\n## VERSION {i}\n{result.strip()}"
            })

            # Add the feedback (as if coming from a user or critic)
            messages.append({
                "role": "assistant",
                "content": f"---\n## FEEDBACK to VERSION {i}\n{feedback.strip()}"
            })

        return messages

    def get_history(self) -> str:
        """
        Get a formatted string representation of the entire history.

        Returns:
            str: A formatted string containing all history entries with results and feedback,
                 separated by revision numbers and dividers.
        """
        if len(self.data)==0:
            return ""
        history = []
        for i in range(len(self.data)):
            result, feedback = self.get_history_by_index(i)
            history.append(
                f"{'-'*3}\n"
                f"## **revision {i}**\n"
                f"{result}\n\n"
                f"## **Feedbacks to the revision\n{feedback}**\n\n"
            )

        history.append('-' * 3)
        return "\n".join(history)
            


    def __str__(self) -> str:
        """
        String representation of the history.

        Returns:
            str: Formatted string of the entire history.
        """
        return self.get_history()
    
    

    def to_dict(self) -> dict:
        """
        Convert the history to a dictionary format.

        Returns:
            dict: A dictionary containing the history with revision numbers,
                 results, and feedback for each entry.
        """
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
    


def test_get_history():
    history = History()
    
    # Test empty history
    print("Empty history output:")
    print(history.get_history())

    # Add entries
    history.save("Result 1", "Feedback 1")
    history.save("Result 2", "Feedback 2")
    history.save("Result 3", "Feedback 3")
 
    
    print("History output after adding entries:")
    print(history.get_history())


if __name__ == "__main__":
    test_get_history()