from sqlalchemy.orm import sessionmaker
from models import init_db
from mistral_trainer import MistralMinesweeperTrainer
from evaluator import MinesweeperEvaluator
import os
from dotenv import load_dotenv
import time

load_dotenv()

def evaluate_and_train(iteration: int = None):
    # Initialize database
    engine = init_db()
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create trainer and evaluator
    trainer = MistralMinesweeperTrainer(session)
    evaluator = MinesweeperEvaluator(session, trainer.model_name)

    # Evaluate current model
    print(f"\nEvaluating model (iteration {iteration if iteration else 'initial'})...")
    results = evaluator.evaluate_model(num_games=50)
    evaluator.print_evaluation_results(results)

    # Generate new training data
    print("\nGenerating new training data...")
    trainer.generate_training_data(num_games=100)

    # Prepare data for fine-tuning
    print("Preparing data for fine-tuning...")
    training_data = trainer.prepare_training_dataset(iteration)

    # Fine-tune the model
    print("Starting fine-tuning process...")
    model_path = trainer.fine_tune_model(training_data, iteration)
    print(f"Fine-tuning completed. Model saved to: {model_path}")

    # Load the fine-tuned model
    trainer.load_model(model_path)

def main():
    # Initial evaluation and training
    evaluate_and_train()

    while True:
        response = input("\nWould you like to perform another training iteration? (y/n): ")
        if response.lower() != 'y':
            break
        
        iteration = int(input("Enter iteration number: "))
        evaluate_and_train(iteration)

if __name__ == "__main__":
    main() 