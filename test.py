import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import os
from utils import calculate_metrics, seeding
from model import build_resunetplusplus
from metrics import DiceBCELoss, DiceLoss
from train import DATASET, DataLoader, load_data

def load_model(checkpoint_path, model, device):
    model.load_state_dict(torch.load(checkpoint_path))  # Modelni saqlangan checkpoint'dan yuklaydi
    model = model.to(device)  # Modelni tanlangan qurilmaga (CPU yoki GPU) o'tkazadi
    model.eval()  # Modelni baholash rejimiga o'tkazadi
    return model


def visualize_results(images, masks, preds, idx=0):
    image = images[idx]  # Indeks bo'yicha rasmni tanlaydi
    mask = masks[idx]  # Indeks bo'yicha haqiqiy maskani tanlaydi
    pred = preds[idx]  # Indeks bo'yicha bashorat qilingan maskani tanlaydi

    image = np.transpose(image, (1, 2, 0))  # Rasmni (C, H, W) formatidan (H, W, C) formatiga o'tkazadi
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=0)  # Agar maskaning o'lchami 3D bo'lsa, uni 2D ga o'tkazadi
    if pred.ndim == 3:
        pred = np.squeeze(pred, axis=0)  # Agar bashorat qilingan maskaning o'lchami 3D bo'lsa, uni 2D ga o'tkazadi

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # Uchta subplots yaratadi
    ax[0].imshow(image)  # Original rasmini ko'rsatadi
    ax[0].set_title('Original Image')  # Title qo'shadi
    ax[1].imshow(mask, cmap='gray')  # Haqiqiy maskani ko'rsatadi, rang sxemasi qora va oq
    ax[1].set_title('Ground Truth Mask')  # Title qo'shadi
    ax[2].imshow(pred, cmap='gray')  # Bashorat qilingan maskani ko'rsatadi, rang sxemasi qora va oq
    ax[2].set_title('Predicted Mask')  # Title qo'shadi
    plt.show()  # Grafikni ko'rsatadi


def test_model(model, loader, loss_fn, device):
    model.eval()  # Modelni baholash rejimiga o'tkazadi
    test_loss = 0.0
    test_jac = 0.0
    test_f1 = 0.0
    test_recall = 0.0
    test_precision = 0.0

    images, masks, preds = [], [], []
    with torch.no_grad():  # Gradient hisoblashni o'chiradi
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)  # Kirish ma'lumotlarini tanlangan qurilmaga o'tkazadi
            y = y.to(device, dtype=torch.float32)  # Haqiqiy maskani tanlangan qurilmaga o'tkazadi

            y_pred = model(x)  # Modeldan bashorat qilingan maskani olish
            loss = loss_fn(y_pred, y)  # Yo'qotishni hisoblash
            test_loss += loss.item()  # Yo'qotishni qo'shadi

            y_pred = torch.sigmoid(y_pred)  # Sigmoid funksiyasini qo'llaydi

            x_cpu = x.cpu().numpy()  # Kirish ma'lumotlarini CPU ga ko'chiradi va numpy massivga aylantiradi
            y_cpu = y.cpu().numpy()  # Haqiqiy maskani CPU ga ko'chiradi va numpy massivga aylantiradi
            y_pred_cpu = y_pred.cpu().numpy()  # Bashorat qilingan maskani CPU ga ko'chiradi va numpy massivga aylantiradi

            images.extend(x_cpu)  # Kirish rasmlarini ro'yxatga qo'shadi
            masks.extend(y_cpu)  # Haqiqiy maskalarni ro'yxatga qo'shadi
            preds.extend(y_pred_cpu)  # Bashorat qilingan maskalarni ro'yxatga qo'shadi

            batch_jac, batch_f1, batch_recall, batch_precision = [], [], [], []
            for yt, yp in zip(y, y_pred):
                score = calculate_metrics(yt, yp)  # Metrikalarni hisoblaydi
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            test_jac += np.mean(batch_jac)  # Jaccard ko'rsatkichini qo'shadi
            test_f1 += np.mean(batch_f1)  # F1 ko'rsatkichini qo'shadi
            test_recall += np.mean(batch_recall)  # Recall ko'rsatkichini qo'shadi
            test_precision += np.mean(batch_precision)  # Precision ko'rsatkichini qo'shadi

    test_loss /= len(loader)  # O'rtacha yo'qotishni hisoblaydi
    test_jac /= len(loader)  # O'rtacha Jaccard ko'rsatkichini hisoblaydi
    test_f1 /= len(loader)  # O'rtacha F1 ko'rsatkichini hisoblaydi
    test_recall /= len(loader)  # O'rtacha Recall ko'rsatkichini hisoblaydi
    test_precision /= len(loader)  # O'rtacha Precision ko'rsatkichini hisoblaydi

    visualize_results(images, masks, preds)  # Natijalarni vizualizatsiya qiladi

    print(f"Test Loss: {test_loss:.4f}")  # Test yo'qotishni chiqaradi
    print(f"Test Metrics:\nJaccard: {test_jac:.4f} - F1: {test_f1:.4f} - Recall: {test_recall:.4f} - Precision: {test_precision:.4f}")  # Test ko'rsatkichlarini chiqaradi


if __name__ == "__main__":    
    seeding(42)  # Tasodifiy sonlar uchun urug'larni o'rnatadi

    checkpoint_path = ""  # Modelni yuklash uchun checkpoint yo'li
    path = ""  # Ma'lumotlarni yuklash uchun yo'l
    batch_size = 4  # Batch hajmi
    image_size = 512  # Rasm o'lchami
    size = (image_size, image_size)  # Rasm o'lchamining tuple'i

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)  # Ma'lumotlarni yuklaydi

    test_dataset = DATASET(test_x, test_y, size)  # Test datasetini yaratadi
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )  # Test dataloader'ini yaratadi

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU yoki CPU tanlaydi
    model = build_resunetplusplus()  # Modelni quradi
    model = load_model(checkpoint_path, model, device)  # Modelni yuklaydi
    loss_fn = DiceBCELoss()  # Yo'qotish funksiyasini yaratadi
    test_model(model, test_loader, loss_fn, device)  # Modelni sinaydi
