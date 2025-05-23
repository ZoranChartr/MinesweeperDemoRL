from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import json

Base = declarative_base()

class GameState(Base):
    __tablename__ = 'game_states'
    
    id = Column(Integer, primary_key=True)
    board_size = Column(Integer)  # Size of the board (e.g., 9 for 9x9)
    mine_count = Column(Integer)  # Number of mines
    board_state = Column(JSON)    # Serialized board state
    mine_locations = Column(JSON) # Serialized mine locations
    created_at = Column(DateTime, default=datetime.utcnow)
    
    actions = relationship("Action", back_populates="game_state")

class Action(Base):
    __tablename__ = 'actions'
    
    id = Column(Integer, primary_key=True)
    game_state_id = Column(Integer, ForeignKey('game_states.id'))
    action_type = Column(String)  # 'click' or 'flag'
    x = Column(Integer)           # x coordinate
    y = Column(Integer)           # y coordinate
    outcome = Column(String)      # 'success', 'mine', 'invalid'
    reward = Column(Float)        # Reward value for reinforcement learning
    created_at = Column(DateTime, default=datetime.utcnow)
    
    game_state = relationship("GameState", back_populates="actions")

class TrainingData(Base):
    __tablename__ = 'training_data'
    
    id = Column(Integer, primary_key=True)
    input_data = Column(JSON)     # Serialized input for the model
    output_data = Column(JSON)    # Serialized expected output
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db(database_url="sqlite:///minesweeper.db"):
    """Initialize the database and create tables."""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return engine 