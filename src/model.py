import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Bloque de doble convolución que se repite en la U-Net"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleUNet, self).__init__()
        
        # encoder (Contracción)
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # El cuello de la red (Bottleneck)
        self.bottleneck = DoubleConv(128, 256)
        
        # Decoder (Expansión)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128) # 256 por la concatenación (skip connection)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        
        # Capa final de salida (predicción por píxel)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Camino de bajada
        x1 = self.down1(x)
        x2 = self.pool1(x1)
        x3 = self.down2(x2)
        x4 = self.pool2(x3)
        
        # Cuello
        b = self.bottleneck(x4)
        
        # Camino de subida con Skip Connections
        u2 = self.up2(b)
        u2 = torch.cat([u2, x3], dim=1) # Concatenamos la pista del encoder
        u2 = self.conv_up2(u2)
        
        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.conv_up1(u1)
        
        return self.final_conv(u1)

# Pequeña prueba para verificar que las dimensiones coincidan
if __name__ == "__main__":
    model = SimpleUNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256) # Imagen simulada de 256x256 píxeles
    y = model(x)
    print("Dimensiones de salida:", y.shape) # Debería ser [1, 1, 256, 256]
