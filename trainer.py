import os
from typing import List, Dict, Tuple
import json
from datetime import datetime
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from models import GameState, Action, TrainingData
from minesweeper import Minesweeper

load_dotenv()

class MinesweeperTrainer:
    def __init__(self, session: Session):
        self.session = session
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.base_model = "gpt-4.1-nano-2025-04-14"  # Using GPT-4.1 nano
        self.current_model = self.base_model
        self.training_iteration = 0

    def generate_training_game(self) -> Tuple[GameState, List[Action]]:
        """Generate a single training game and record all actions."""
        game = Minesweeper(size=9, mine_count=10)
        game_state = GameState(
            board_size=game.size,
            mine_count=game.mine_count,
            board_state=game.get_state(),
            mine_locations=game.mines.tolist()
        )
        self.session.add(game_state)
        self.session.commit()

        actions = []
        while not game.game_over and not game.won:
            # Get model's prediction
            state_str = game.format_for_model()
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[
                    {"role": "system", "content": "You are a Minesweeper AI. Analyze the board and make the best move."},
                    {"role": "user", "content": state_str}
                ],
                temperature=0.7
            )
            
            # Parse model's response and make move
            try:
                move = self._parse_model_response(response.choices[0].message.content)
                if move:
                    action_type, x, y = move
                    if action_type == 'click':
                        hit_mine, _ = game.reveal_cell(x, y)
                        outcome = 'mine' if hit_mine else 'success'
                        reward = -1.0 if hit_mine else 0.1
                    else:  # flag
                        success = game.toggle_flag(x, y)
                        outcome = 'success' if success else 'invalid'
                        reward = 0.05 if success else -0.1

                    action = Action(
                        game_state_id=game_state.id,
                        action_type=action_type,
                        x=x,
                        y=y,
                        outcome=outcome,
                        reward=reward
                    )
                    actions.append(action)
                    self.session.add(action)
                    self.session.commit()

            except Exception as e:
                print(f"Error processing model response: {e}")
                continue

        return game_state, actions

    def _parse_model_response(self, response: str) -> Tuple[str, int, int]:
        """Parse the model's response to extract the move."""
        # Expected format: "click x y" or "flag x y"
        try:
            parts = response.lower().strip().split()
            if len(parts) != 3:
                return None
            action_type, x, y = parts
            if action_type not in ['click', 'flag']:
                return None
            return action_type, int(x), int(y)
        except:
            return None

    def generate_training_data(self, num_games: int = 100):
        """Generate multiple training games and store the data."""
        for _ in range(num_games):
            game_state, actions = self.generate_training_game()
            
            # Create training data entries
            for action in actions:
                training_data = TrainingData(
                    input_data={
                        'board_state': game_state.board_state,
                        'action_type': action.action_type,
                        'x': action.x,
                        'y': action.y
                    },
                    output_data={
                        'outcome': action.outcome,
                        'reward': action.reward
                    }
                )
                self.session.add(training_data)
            
            self.session.commit()

    def prepare_fine_tuning_data(self, iteration: int = None) -> List[Dict]:
        """Prepare the collected data for fine-tuning."""
        query = self.session.query(TrainingData)
        if iteration is not None:
            # Get only data from the specified iteration
            query = query.filter(TrainingData.created_at >= datetime.utcnow())
        
        training_data = query.all()
        formatted_data = []
        
        for data in training_data:
            formatted_data.append({
                "messages": [
                    {"role": "system", "content": "You are a Minesweeper AI. Analyze the board and make the best move."},
                    {"role": "user", "content": json.dumps(data.input_data)},
                    {"role": "assistant", "content": json.dumps(data.output_data)}
                ]
            })
        
        return formatted_data

    def fine_tune_model(self, training_data: List[Dict], iteration: int = None):
        """Fine-tune the model with the collected data."""
        # Save training data to file
        filename = f'training_data_{iteration}.jsonl' if iteration is not None else 'training_data.jsonl'
        with open(filename, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')

        # Create fine-tuning job
        response = self.client.fine_tuning.jobs.create(
            training_file=filename,
            model=self.base_model
        )
        
        self.training_iteration += 1
        return response.id

    def update_model(self, fine_tuned_model: str):
        """Update the current model to use the fine-tuned version."""
        self.current_model = fine_tuned_model

    def get_training_status(self, job_id: str) -> Dict:
        """Get the status of a fine-tuning job."""
        return self.client.fine_tuning.jobs.retrieve(job_id)

    def list_fine_tuned_models(self) -> List[Dict]:
        """List all fine-tuned models."""
        return self.client.fine_tuning.jobs.list() 