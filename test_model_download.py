from sqlalchemy.orm import sessionmaker
from models import init_db
from mistral_trainer import MistralMinesweeperTrainer
import os

def test_model_download():
    print("Initializing database...")
    engine = init_db()
    Session = sessionmaker(bind=engine)
    session = Session()

    print("\nInitializing Mistral trainer (this will download the model)...")
    try:
        trainer = MistralMinesweeperTrainer(session)
        print("\nModel downloaded successfully!")
        
        # Test a simple inference
        print("\nTesting model inference with a simple board...")
        test_board = {
            'board': [
                ['?', '?', '?'],
                ['?', '?', '?'],
                ['?', '?', '?']
            ]
        }
        
        move = trainer.generate_move(test_board)
        print(f"Model generated move: {move}")
        
        print("\nVerifying cache directory...")
        cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
        if os.path.exists(cache_dir):
            print(f"Cache directory exists at: {cache_dir}")
            print("Cache contents:")
            for item in os.listdir(cache_dir):
                print(f"- {item}")
        else:
            print("Warning: Cache directory not found!")
            
    except Exception as e:
        print(f"\nError during model initialization: {str(e)}")
        raise

if __name__ == "__main__":
    test_model_download() 