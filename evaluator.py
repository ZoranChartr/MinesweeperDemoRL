from typing import List, Dict, Tuple
import numpy as np
from sqlalchemy.orm import Session
from minesweeper import Minesweeper
from models import GameState, Action
import os
from dotenv import load_dotenv

load_dotenv()

class MinesweeperEvaluator:
    def __init__(self, session: Session, model_name: str):
        self.session = session
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model_name

    def evaluate_model(self, num_games: int = 50) -> Dict:
        """Evaluate the model's performance over multiple games."""
        results = {
            'games_played': 0,
            'games_won': 0,
            'mines_hit': 0,
            'total_moves': 0,
            'successful_moves': 0,
            'invalid_moves': 0
        }

        for _ in range(num_games):
            game = Minesweeper(size=9, mine_count=10)
            game_state = GameState(
                board_size=game.size,
                mine_count=game.mine_count,
                board_state=game.get_state(),
                mine_locations=game.mines.tolist()
            )
            self.session.add(game_state)
            self.session.commit()

            while not game.game_over and not game.won:
                state_str = game.format_for_model()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a Minesweeper AI. Analyze the board and make the best move."},
                        {"role": "user", "content": state_str}
                    ],
                    temperature=0.7
                )

                try:
                    move = self._parse_model_response(response.choices[0].message.content)
                    if move:
                        action_type, x, y = move
                        results['total_moves'] += 1

                        if action_type == 'click':
                            hit_mine, _ = game.reveal_cell(x, y)
                            if hit_mine:
                                results['mines_hit'] += 1
                            else:
                                results['successful_moves'] += 1
                        else:  # flag
                            success = game.toggle_flag(x, y)
                            if success:
                                results['successful_moves'] += 1
                            else:
                                results['invalid_moves'] += 1

                except Exception as e:
                    results['invalid_moves'] += 1
                    continue

            results['games_played'] += 1
            if game.won:
                results['games_won'] += 1

        # Calculate metrics
        results['win_rate'] = results['games_won'] / results['games_played']
        results['success_rate'] = results['successful_moves'] / results['total_moves'] if results['total_moves'] > 0 else 0
        results['mine_hit_rate'] = results['mines_hit'] / results['total_moves'] if results['total_moves'] > 0 else 0

        return results

    def _parse_model_response(self, response: str) -> Tuple[str, int, int]:
        """Parse the model's response to extract the move."""
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

    def print_evaluation_results(self, results: Dict):
        """Print formatted evaluation results."""
        print("\nModel Evaluation Results:")
        print(f"Games Played: {results['games_played']}")
        print(f"Games Won: {results['games_won']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Total Moves: {results['total_moves']}")
        print(f"Successful Moves: {results['successful_moves']}")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Mines Hit: {results['mines_hit']}")
        print(f"Mine Hit Rate: {results['mine_hit_rate']:.2%}")
        print(f"Invalid Moves: {results['invalid_moves']}") 