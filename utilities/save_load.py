import torch
import os
from pathlib import Path

def save_model(model, file_name):
    model_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'models', file_name))
    # print(model_path)
    try:
        torch.save(model.state_dict(), model_path)
        print("Model saving successful")
    except:
        print("Path doesn't exist")

def load_model(init_model, file_name):
    try:
        model_main_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'models', file_name))
        init_model.load_state_dict(torch.load(model_main_path))
        return init_model
    except:
        print("Path doesn't exist")
