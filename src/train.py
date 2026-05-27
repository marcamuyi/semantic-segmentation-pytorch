import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SimpleUNet
from dataset import SemanticSegmentationDataset
from utils import calculate_iou, save_checkpoint

def train_fn():
    # 1. Configuración de hiperparámetros básicos
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    NUM_EPOCHS = 5 # Pocas épocas para probar rápido
    
    # RUTA DE EJEMPLO (Se cambiarán cuando entrenes en Google Colab)
    IMG_DIR = "data/raw/images"
    MASK_DIR = "data/raw/masks"

    # 2. Inicializar Modelo, Pérdida y Optimizado
    model = SimpleUNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # Ideal para segmentación binaria
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Entrenando en el dispositivo: {DEVICE}")
    print("Repositorio listo para recibir datos e iniciar el entrenamiento.")
    
    # Nota: El bucle loop de entrenamiento 'for epoch in range(NUM_EPOCHS):'
    # se ejecutará de forma masiva cuando conectemos esto a Google Colab.

if __name__ == "__main__":
    train_fn()
