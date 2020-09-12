import torch

DATA_DIR = './data'
DEBUG = False
BATCH_SIZE = 64
EPOCHS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DOWNLOAD = True
NUM_WORKERS = 4