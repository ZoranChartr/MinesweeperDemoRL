
# Minesweeper AI Trainer

This project implements a system to train an AI model to play Minesweeper using fine-tuning capabilities. The system generates training data through gameplay, stores it in a database, and uses it to fine-tune a model.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

- `models.py`: Database models for storing game states and training data
- `minesweeper.py`: Minesweeper game engine implementation
- `trainer.py`: Training data generation and model fine-tuning logic
- `main.py`: Main script to run the training process

## Usage

To start the training process:

```bash
python main.py
```

This will:
1. Generate 100 training games
2. Store the game states and actions in the database
3. Prepare the data for fine-tuning
4. Start the fine-tuning process with OpenAI

## How it Works

1. The system generates Minesweeper games and records all actions taken
2. Each game state and action is stored in the database
3. The training data is formatted for OpenAI's fine-tuning API
4. The model is fine-tuned to learn from successful and unsuccessful moves

## Database

The system uses SQLite by default, with the following tables:
- `game_states`: Stores board states and mine locations
- `actions`: Records moves and their outcomes
- `training_data`: Stores formatted data for fine-tuning

## Notes

- The system uses a local LLaMa for fine-tuning
- Training data is stored in `training_data.jsonl` before being sent to OpenAI
- The fine-tuning job ID is printed at the end of the process 

