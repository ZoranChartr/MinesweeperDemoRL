import os
from typing import List, Dict, Tuple
import json
from datetime import datetime
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from models import GameState, Action, TrainingData
from minesweeper import Minesweeper
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
import numpy as np

load_dotenv()

class MistralMinesweeperTrainer:
    def __init__(self, session: Session):
        self.session = session
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set up model cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
        if not os.path.exists(self.cache_dir):
            raise ValueError(f"Model cache directory not found at {self.cache_dir}. Please run download_model.py first.")
        
        # Initialize model and tokenizer from local cache
        self.tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            cache_dir=self.cache_dir,
            local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization for efficient training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load base model from local cache
        self.model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            cache_dir=self.cache_dir,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=True
        )
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            r=8,  # Reduced from 16 to save memory
            lora_alpha=16,  # Reduced from 32 to save memory
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.training_iteration = 0

    def format_prompt(self, game_state: dict) -> str:
        """Format the game state into a prompt for the model."""
        board = game_state['board']
        prompt = "Current Minesweeper board state:\n"
        for row in board:
            prompt += " ".join(str(cell) for cell in row) + "\n"
        prompt += "\nMake your move (format: 'click x y' or 'flag x y'):"
        return prompt

    def generate_move(self, game_state: dict) -> Tuple[str, int, int]:
        """Generate a move using the model."""
        prompt = self.format_prompt(game_state)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_model_response(response)

    def _parse_model_response(self, response: str) -> Tuple[str, int, int]:
        """Parse the model's response to extract the move."""
        try:
            # Extract the last line which should contain the move
            move_line = response.strip().split('\n')[-1]
            parts = move_line.lower().strip().split()
            if len(parts) != 3:
                return None
            action_type, x, y = parts
            if action_type not in ['click', 'flag']:
                return None
            return action_type, int(x), int(y)
        except:
            return None

    def generate_training_data(self, num_games: int = 100):
        """Generate training data and store in database."""
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
                move = self.generate_move(game.get_state())
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
                    self.session.add(action)
                    self.session.commit()

    def prepare_training_dataset(self, iteration: int = None) -> Dict:
        """Prepare the collected data for fine-tuning."""
        query = self.session.query(TrainingData)
        if iteration is not None:
            query = query.filter(TrainingData.created_at >= datetime.utcnow())
        
        training_data = query.all()
        
        # Format data for training
        prompts = []
        responses = []
        
        for data in training_data:
            prompt = self.format_prompt(data.input_data['board_state'])
            response = f"{data.input_data['action_type']} {data.input_data['x']} {data.input_data['y']}"
            
            prompts.append(prompt)
            responses.append(response)
        
        return {
            'prompts': prompts,
            'responses': responses
        }

    def fine_tune_model(self, training_data: Dict, iteration: int = None):
        """Fine-tune the model using the collected data."""
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=f"tinyllama_minesweeper_{iteration if iteration else 'base'}",
            num_train_epochs=3,
            per_device_train_batch_size=2,  # Reduced batch size for smaller GPU
            gradient_accumulation_steps=8,  # Increased to compensate for smaller batch size
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch"
        )
        
        # Create dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['prompts'],
                examples['responses'],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        
        # Train the model
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenize_function(training_data),
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        
        # Save the model
        output_dir = f"tinyllama_minesweeper_{iteration if iteration else 'base'}"
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        self.training_iteration += 1
        return output_dir

    def load_model(self, model_path: str):
        """Load a fine-tuned model."""
        cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=True  # Use cached version
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            local_files_only=True  # Use cached version
        ) 