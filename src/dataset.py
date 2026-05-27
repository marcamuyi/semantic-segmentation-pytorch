import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Ruta a la carpeta con las imágenes originales.
            mask_dir (str): Ruta a la carpeta con las máscaras (ground truth).
            transform (albumentations.Compose): Transformaciones/Aumentación de datos.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Listamos y ordenamos los archivos para que coincidan perfectamente
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 1. Obtener las rutas de la imagen y su máscara correspondiente
        img_path = os.path.join(self.image_dir, self.images[index])
        # Asumimos que la máscara tiene el mismo nombre que la imagen
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        # 2. Leer la imagen (en color) y la máscara (en escala de grises)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convertir a RGB estándar
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Las máscaras suelen ser de 1 solo canal
        
        # 3. Aplicar transformaciones (como redimensionar o aumentar datos)
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
        # 4. Convertir a tensores de PyTorch manualmente si no se hace en las transformaciones
        # Cambiamos dimensiones de (H, W, C) a (C, H, W) que es lo que pide PyTorch
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        
        return torch.tensor(image), torch.tensor(mask)
