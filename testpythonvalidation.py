from pythonvalidator import pythonValidator
import asyncio

python_code = '''python
import tkinter as tk
import random

# Function to evaluate the game result
def evaluate_game_result(player_choice, ai_choice):
    if player_choice == ai_choice:
        return "Draw"
    elif (player_choice == "Rock" and ai_choice == "Scissors") or \
         (player_choice == "Paper" and ai_choice == "Rock") or \
         (player_choice == "Scissors" and ai_choice == "Paper"):
        return "Win"
    else:
        return "Lose"

# Function to update the display with results
def update_display(player_choice, ai_choice, result):
    player_choice_label.config(text=f"Your Choice: {player_choice}")
    ai_choice_label.config(text=f"AI Choice: {ai_choice}")
    result_label.config(text=f"Result: {result}")

    if result == "Win":
        result_label.config(fg="green")
    elif result == "Lose":
        result_label.config(fg="red")
    else:
        result_label.config(fg="yellow")

# Function to handle player's choice
def on_choice_selected(choice):
    ai_choice = random.choice(["Rock", "Paper", "Scissors"])
    game_result = evaluate_game_result(choice, ai_choice)
    update_display(choice, ai_choice, game_result)

# Function to reset the game
def reset_game():
    player_choice_label.config(text="Your Choice: None")
    ai_choice_label.config(text="AI Choice: None")
    result_label.config(text="Result: Awaiting...", fg="black")

# Function to exit the game
def exit_game():
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Rock-Paper-Scissors")
root.geometry("600x400")

# Title label
title_label = tk.Label(root, text="Rock-Paper-Scissors", font=("Arial", 24))
title_label.pack(pady=20)

# Player choice buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

rock_button = tk.Button(button_frame, text="Rock", command=lambda: on_choice_selected("Rock"), width=10)
paper_button = tk.Button(button_frame, text="Paper", command=lambda: on_choice_selected("Paper"), width=10)
scissors_button = tk.Button(button_frame, text="Scissors", command=lambda: on_choice_selected("Scissors"), width=10)

rock_button.pack(side=tk.LEFT, padx=10)
paper_button.pack(side=tk.LEFT, padx=10)
scissors_button.pack(side=tk.LEFT, padx=10)

# Display labels
player_choice_label = tk.Label(root, text="Your Choice: None", font=("Arial", 20))
player_choice_label.pack(pady=10)

ai_choice_label = tk.Label(root, text="AI Choice: None", font=("Arial", 20))
ai_choice_label.pack(pady=10)

result_label = tk.Label(root, text="Result: Awaiting...", font=("Arial", 24))
result_label.pack(pady=10)

# Play Again and Exit buttons
play_again_button = tk.Button(root, text="Play Again", command=reset_game, width=10)
exit_button = tk.Button(root, text="Exit", command=exit_game, width=10)

play_again_button.pack(side=tk.LEFT, padx=20, pady=20)
exit_button.pack(side=tk.RIGHT, padx=20, pady=20)

# Start the GUI loop
root.mainloop()
'''

task_obj = '''
Develop a Rock-Paper-Scissors game with a graphical user interface (GUI) in Python. 
    The game should allow a player to compete against a naive AI that randomly selects Rock, Paper, or Scissors. 
    The UI should display the player's choice, the AI's choice, and the game result (win, lose, or draw). 
    Provide an interactive and user-friendly experience.
'''

async def main():
    python_validator = pythonValidator()
    result, status, test_code, runresult = await python_validator.validate(task_obj, python_code, '')
    print(result)
    print(status)
    print(test_code)
    print(f'runresult: {runresult}')

if __name__ == "__main__":
    asyncio.run(main())

