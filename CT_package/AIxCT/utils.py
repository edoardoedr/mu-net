import numpy as np
from PIL import Image, ImageFilter, ImageOps, ExifTags
import statistics as st
from multiprocessing import Pool, cpu_count
import time
from functools import partial
import os
from torchvision import transforms
import torch.nn.functional as F
import torch

def iou2D(pred, target, n_classes = 3):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(0, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union > 0:
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious)

def calcola_moda(array):
    zeta = array.shape[2]
    rig = array.shape[0]
    col = array.shape[1]
    moda = np.empty(zeta, dtype=np.uint8)
    new_array = np.empty([rig, col], dtype=np.uint8)

    for r in range(rig):
        for c in range(col):
            moda = np.copy(array[r, c, :])
            new_array[r, c] = np.copy(st.mode(moda))

    return new_array

def calcola_moda_parallelized(stacks):
    start_time = time.perf_counter()
    z = stacks[0].shape[2]
    rig = stacks[0].shape[0]
    col = stacks[0].shape[1]
    num_stack = len(stacks)
    pila = np.empty([rig, col, num_stack], dtype=np.uint8)
    new_stack = np.empty([rig, col, z], dtype=np.uint8)

    lista_a3 = []

    for x in range(0, z, 1):
        immagini = []
        for i in range(len(stacks)):
            stac = stacks[i]
            immagini.append(stac[:, :, x])
        pila = np.dstack([immagine for immagine in immagini])
        # print(pila.shape)
        lista_a3.append(pila)

    with Pool() as pool:
        result = pool.map(calcola_moda, lista_a3)

    for n in range(z):
        new_stack[:, :, n] = np.copy(result[n])

    finish_time = time.perf_counter()
    print("Moda calcolata in {} seconds - using multiprocessing".format(finish_time - start_time))
    print("---")

    return new_stack

def return_output_dir(directory_principale, nome ):
    cont = 0
    trovata = True
    while(trovata == True):
        nome_cartella = "output_" + nome + f"_{cont:02d}"
        # Crea il percorso completo della cartella da creare
        percorso_cartella = os.path.join(directory_principale, nome_cartella)
        # Controlla se la cartella esiste già
        if os.path.exists(percorso_cartella):
            trovata = True
            cont = cont + 1
        else:
            os.mkdir(percorso_cartella)
            trovata = False
    percorso_cartella = percorso_cartella + "/"
    
    return percorso_cartella

def unfold_image(image, tiles_dim, patches = True):

    if image.dtype == np.uint16:
        img_np = np.copy(image)
        image = img_np.astype(np.float32)/65535

    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)

    if patches == True:

        H, W = image.shape[-2:]
        n_H = (H + tiles_dim - 1) // tiles_dim
        n_W = (W + tiles_dim - 1) // tiles_dim

        pad_H = (tiles_dim - H % tiles_dim) % tiles_dim
        pad_W = (tiles_dim - W % tiles_dim) % tiles_dim

        image = F.pad(image, (0, pad_W, 0, pad_H), value=0)

        # Scompongo l'immagine in immagini di dimensione [1,1,k,k] usando la funzione unfold
        img_unfold = image.unfold(-2, tiles_dim, tiles_dim).unfold(-2, tiles_dim, tiles_dim)

        # Trasformo le immagini scomposte in un tensore di dimensione [n_H * n_W, 1, k, k]
        img_unfold = img_unfold.contiguous().view(-1, 1, tiles_dim, tiles_dim)

    else:
        img_unfold = image

    return img_unfold

def fold_image(img_unfold, H, W, tiles_dim):

    n_H = (H + tiles_dim - 1) // tiles_dim
    n_W = (W + tiles_dim - 1) // tiles_dim

    pad_H = (tiles_dim - H % tiles_dim) % tiles_dim
    pad_W = (tiles_dim - W % tiles_dim) % tiles_dim


    img_recon = torch.zeros(1, 1, H + pad_H, W + pad_W)

    # Scorro il tensore di dimensione [n_H * n_W, 1, k, k] con un ciclo for
    for i in range(n_H * n_W):
        # Calcolo le coordinate della posizione del piccolo tensore nell'immagine grande
        h = (i // n_W) * tiles_dim
        w = (i % n_W) * tiles_dim

        # Copio il piccolo tensore nell'immagine grande
        img_recon[:, :, h:h+tiles_dim, w:w+tiles_dim] = img_unfold[i]

    # Rimuovo il padding che ho aggiunto
    image_rc = img_recon[:, :, :H, :W]
    
    return image_rc


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = mask

    def __call__(self, model_output):
        output_modello = model_output[0, self.category, :, : ]
        output_modello = output_modello
        return (output_modello * self.mask).sum()