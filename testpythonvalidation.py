from pythonvalidator import pythonValidator
import asyncio
python_code = '''python
import tkinter as tk
from tkinter import messagebox
import random

# Function to determine the winner of the game
def determine_winner(player_choice, ai_choice):
    if player_choice == ai_choice:
        return "Draw"
    elif (player_choice == "Rock" and ai_choice == "Scissors") or \
         (player_choice == "Scissors" and ai_choice == "Paper") or \
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

if __name__ == "__main__":
    main()
'''

task_obj = '''
Develop a Rock-Paper-Scissors game with a graphical user interface (GUI) in Python. 
    The game should allow a player to compete against a naive AI that randomly selects Rock, Paper, or Scissors. 
    The UI should display the player's choice, the AI's choice, and the game result (win, lose, or draw). 
    Provide an interactive and user-friendly experience.
'''

async def main():
    python_validator = pythonValidator()
    result, status, test_code = await python_validator.validate(task_obj, python_code, '')
    print(result)
    print(status)
    print(test_code)

if __name__ == "__main__":
    asyncio.run(main())

