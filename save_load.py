import torch
import os

def save_model(model, file_name):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', file_name)
    try:
        torch.save(model.state_dict(), model_path)
        print("Model saving successful")
    except:
        print("Path doesn't exist")

def load_model(init_model, model_path):
    try:
        init_model.load_state_dict(torch.load(model_path))
        return init_model
    except:
        print("Path doesn't exist")
