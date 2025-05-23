import numpy as np
from typing import Tuple, List, Optional
import json

class Minesweeper:
    def __init__(self, size: int = 9, mine_count: int = 10):
        self.size = size
        self.mine_count = mine_count
        self.board = np.full((size, size), -1)  # -1 represents hidden cells
        self.mines = np.zeros((size, size), dtype=bool)
        self.flags = np.zeros((size, size), dtype=bool)
        self.game_over = False
        self.won = False
        self.place_mines()

    def place_mines(self):
        """Randomly place mines on the board."""
        positions = np.random.choice(self.size * self.size, self.mine_count, replace=False)
        for pos in positions:
            x, y = pos // self.size, pos % self.size
            self.mines[x, y] = True

    def count_adjacent_mines(self, x: int, y: int) -> int:
        """Count the number of mines adjacent to a cell."""
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    count += self.mines[nx, ny]
        return count

    def reveal_cell(self, x: int, y: int) -> Tuple[bool, int]:
        """
        Reveal a cell and return (hit_mine, number_of_adjacent_mines).
        Returns (-1, -1) if the cell is already revealed or flagged.
        """
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False, -1
        
        if self.board[x, y] != -1 or self.flags[x, y]:
            return False, -1

        if self.mines[x, y]:
            self.game_over = True
            return True, -1

        adjacent_mines = self.count_adjacent_mines(x, y)
        self.board[x, y] = adjacent_mines

        if adjacent_mines == 0:
            # Reveal adjacent cells for empty cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        self.reveal_cell(nx, ny)

        self.check_win()
        return False, adjacent_mines

    def toggle_flag(self, x: int, y: int) -> bool:
        """Toggle a flag on a cell. Returns True if successful."""
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        if self.board[x, y] != -1:
            return False
        self.flags[x, y] = not self.flags[x, y]
        return True

    def check_win(self):
        """Check if the game is won."""
        hidden_cells = np.sum(self.board == -1)
        if hidden_cells == self.mine_count:
            self.won = True

    def get_state(self) -> dict:
        """Get the current game state as a dictionary."""
        return {
            'board': self.board.tolist(),
            'mines': self.mines.tolist(),
            'flags': self.flags.tolist(),
            'game_over': self.game_over,
            'won': self.won
        }

    def get_visible_state(self) -> np.ndarray:
        """Get the current visible state of the board."""
        visible = self.board.copy()
        visible[self.flags] = 10  # 10 represents flags
        return visible

    def format_for_model(self) -> str:
        """Format the current game state for the model input."""
        visible_state = self.get_visible_state()
        state_str = "Current board state:\n"
        for row in visible_state:
            state_str += " ".join(str(cell) for cell in row) + "\n"
        return state_str

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves (unrevealed cells that aren't flagged)."""
        valid_moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] == -1 and not self.flags[x, y]:
                    valid_moves.append((x, y))
        return valid_moves 