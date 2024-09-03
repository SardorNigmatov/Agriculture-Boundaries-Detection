import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from model import build_resunetplusplus

def load_model():
    model = build_resunetplusplus()
    state_dict = torch.load("/files/checkpoint.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Tasvirni preprocessing qilish
def preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((512, 512)),  # Tasvirni model kirish o'lchamiga moslashtirish
        T.ToTensor(),          # Tasvirni tensor shakliga o'tkazish
    ])
    return transform(image).unsqueeze(0), image  # Batch o'lchamini qo'shish va asl rasmni qaytarish

# Maskada chegara chiziqlarini chizish
def draw_boundaries_on_image(original_image, mask):
    # Predicted maskni ikkilik formatga o'tkazish (thresholding)
    binary_mask = (mask > 0.5).astype(np.uint8)  # 0.5 qiymatidan yuqori bo'lgan ostonaviy

    # Ikkilik maskni original tasvir o'lchamiga moslashtirish
    binary_mask_resized = cv2.resize(binary_mask, (original_image.width, original_image.height), interpolation=cv2.INTER_NEAREST)

    # Ikkilik maskada konturlarni topish
    contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Original tasvirni numpy arrayga o'tkazish
    image_with_boundaries = np.array(original_image)

    # Chegaralarni tasvirga chizish
    for contour in contours:
        cv2.drawContours(image_with_boundaries, [contour], -1, (0, 0, 255), 2)  # (0, 0, 255) - qizil rang

    return image_with_boundaries

# Modelni yuklab olish va tasvirni preprocess qilish
image_path = ['img0.jpg','img1.png','img4.jpg','img7.jpg','img8.jpg','img9.jpg','img10.jpg','img11.jpg','img12.jpg','img13.jpg']
for i in range(len(image_path)):
    input_image, original_image = preprocess(image_path[i])

    # Model orqali taxmin qilish
    with torch.no_grad():
        model = load_model()
        output = model(input_image)
        predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy()

    # Tasvirda chegara chizish
    image_with_boundaries = draw_boundaries_on_image(original_image, predicted_mask)

    # Natijalarni ko'rsatish
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Kirish Tasvir')
    plt.imshow(original_image)

    plt.subplot(1, 3, 2)
    plt.title('Taxmin Qilingan Mask')
    plt.imshow(predicted_mask, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Chegaralari Chizilgan Tasvir')
    plt.imshow(image_with_boundaries)
    plt.savefig(f"{i}_results.png")
