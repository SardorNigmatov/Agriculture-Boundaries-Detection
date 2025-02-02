import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff

########################### Yo'qotish funksiyalari (Loss Functions) ###########################
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Sigmoid orqali chiqishni (inputs) aktivatsiya qiladi
        inputs = torch.sigmoid(inputs)

        # Inputs va targets o'lchamlarini bir tekislikka keltiradi
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Dice ko'rsatkichi uchun o'zaro kesishgan qismni hisoblaydi
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        # Dice yo'qotish funksiyasi, uni minimalizatsiya qilish kerak
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Sigmoid orqali chiqishni (inputs) aktivatsiya qiladi
        inputs = torch.sigmoid(inputs)

        # Inputs va targets o'lchamlarini bir tekislikka keltiradi
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Dice yo'qotish funksiyasini hisoblaydi
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        # Binary Cross Entropy (BCE) yo'qotish funksiyasi
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # Dice va BCE yo'qotish funksiyalarining yig'indisi
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

###########################  O'lchovlar (Metrics)  ###########################

def precision(y_true, y_pred):
    # Aniqlikni (Precision) hisoblaydi: y_pred bilan y_true o'rtasidagi kesishishni topadi
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    # Eslash (Recall) hisoblaydi: y_true bilan y_pred o'rtasidagi kesishishni topadi
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2(y_true, y_pred, beta=2):
    # F2 ko'rsatkichi, precision va recall orqali hisoblanadi
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    # Dice ko'rsatkichi hisoblaydi: 2 marta kesishishni (intersection) soniga bo'linadi
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    # Jaccard indeksi (IoU): y_true va y_pred o'rtasidagi kesishish va yig'ish
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def hd_dist(preds, targets):
    # Hausdorff masofasi: segmentatsiya chegaralari orasidagi maksimal masofa
    haussdorf_dist = directed_hausdorff(preds, targets)[0]
    return haussdorf_dist
