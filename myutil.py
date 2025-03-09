import ast

def comment_out_non_functions_and_imports(code: str) -> str:
    """
    Processes the input Python source code and comments out all lines except those that belong to:
      - Function definitions (including their entire bodies)
      - Import statements (both 'import' and 'from ... import ...')
    
    Args:
        code: A string containing the full Python source code.
        
    Returns:
        A new string where any line not part of a function definition or an import statement is commented out.
    """
    try:
        tree = ast.parse(code)
    except Exception as e:
        raise ValueError("Error parsing code") from e

    # Set to keep track of line numbers that should be preserved.
    protected_lines = set()

    def mark_node(node):
        # Use node.lineno and node.end_lineno to mark all lines in this node as protected.
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            for lineno in range(node.lineno, node.end_lineno + 1):
                protected_lines.add(lineno)

    # Walk the AST and mark lines for functions and imports.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)):
            mark_node(node)

    # Process the code line by line.
    new_lines = []
    for idx, line in enumerate(code.splitlines(), start=1):
        # If the line is protected or empty, leave it as is.
        if idx in protected_lines or not line.strip():
            new_lines.append(line)
        else:
            # Prepend a comment marker.
            new_lines.append("# " + line)
    return "\n".join(new_lines)


# Example usage:
if __name__ == "__main__":
    sample_code = '''import tkinter as tk
from tkinter import messagebox
import random

# Function to determine the winner of the game
def determine_winner(player_choice, ai_choice):
    if player_choice == ai_choice:
        return "Draw"
    elif (player_choice == "Rock" and ai_choice == "Scissors") or \\
         (player_choice == "Scissors" and ai_choice == "Paper") or \\
         (player_choice == "Paper" and ai_choice == "Rock"):
        return "You Win!"
    else:
        return "You Lose!"

# Function to generate AI's choice
def ai_select():
    return random.choice(["Rock", "Paper", "Scissors"])

# Function to handle user choice
def user_choice(choice):
    ai_choice = ai_select()
    result = determine_winner(choice, ai_choice)

    # Update GUI with choices and result
    player_choice_label.config(text=f"Your Choice: {choice}")
    ai_choice_label.config(text=f"AI Choice: {ai_choice}")
    result_label.config(text=f"Result: {result}")

# Function to reset the game
def reset_game():
    player_choice_label.config(text="Your Choice: ")
    ai_choice_label.config(text="AI Choice: ")
    result_label.config(text="Result: ")

# Main function to initialize the GUI
def main():
    global root, player_choice_label, ai_choice_label, result_label

    # Create the main window
    root = tk.Tk()
    root.title("Rock-Paper-Scissors Game")

    # Create choice buttons and bind to user_choice function
    rock_button = tk.Button(root, text="Rock", command=lambda: user_choice("Rock"))
    rock_button.pack(pady=10)

    paper_button = tk.Button(root, text="Paper", command=lambda: user_choice("Paper"))
    paper_button.pack(pady=10)

    scissors_button = tk.Button(root, text="Scissors", command=lambda: user_choice("Scissors"))
    scissors_button.pack(pady=10)

    # Create labels for displaying player choice, AI choice, and result
    player_choice_label = tk.Label(root, text="Your Choice: ")
    player_choice_label.pack(pady=5)

    ai_choice_label = tk.Label(root, text="AI Choice: ")
    ai_choice_label.pack(pady=5)

    result_label = tk.Label(root, text="Result: ")
    result_label.pack(pady=5)

    # Create a reset button
    reset_button = tk.Button(root, text="Reset Game", command=reset_game)
    reset_button.pack(pady=20)

    # Start the main loop
    root.mainloop()

python_validator = pythonValidator()
result, status, test_code = await python_validator.validate(task_obj, python_code, '')
async def main():

    print(result)
    print(status)
    print(test_code)

if __name__ == "__main__":
    asyncio.run(main())'''
    
    processed_code = comment_out_non_functions_and_imports(sample_code)
    print(processed_code)
