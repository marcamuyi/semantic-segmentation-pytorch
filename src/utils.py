import torch

def calculate_iou(preds, masks, smooth=1e-6):
    """
    Calcula el Intersection over Union (IoU) para evaluar la segmentación.
    """
    # Convertir predicciones a binario (0 o 1) usando un umbral de 0.5
    preds = (torch.sigmoid(preds) > 0.5).float()
    
    # Aplanar las matrices para operar sobre todos los píxeles juntos
    preds = preds.view(-1)
    masks = masks.view(-1)
    
    intersection = (preds * masks).sum()
    total = preds.sum() + masks.sum()
    union = total - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def save_checkpoint(state, filename="weights/checkpoint.pth"):
    """Guarda los pesos del modelo en la carpeta weights"""
    print("=> Guardando checkpoint...")
    torch.save(state, filename)
