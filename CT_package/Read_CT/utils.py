import numpy as np
import cv2
import os
from PIL import Image, ImageFilter, ExifTags
import statistics as st
from scipy import ndimage
import random
from multiprocessing import Pool
import time
from sklearn.model_selection import train_test_split


def filtro_mode(filter_size, immagine_np):
    immagine = Image.fromarray(immagine_np)
    filtrata = immagine.filter(ImageFilter.ModeFilter(size=filter_size))

    return np.copy(filtrata)

def FiltroUnsharp(immagine):
    immagine = Image.fromarray(immagine)
    immagine_filtrata = immagine.filter(ImageFilter.UnsharpMask(radius=6, percent=350, threshold=1))

    return np.copy(immagine_filtrata)

def fill_holed2D(value_to_fill, structure, immagine_np):
    imm_original = np.copy(immagine_np)
    for value in value_to_fill:
        imm_value = np.copy(immagine_np)
        imm_value[imm_value != value] = 0
        imm_value[imm_value == value] = 1
        imm_value = ndimage.binary_fill_holes(imm_value, structure=structure).astype(np.uint8)
        imm_value[imm_value == 1] = value
        imm_original[imm_value == value] = value
    return imm_original

def Filtrosharpen(immagine):
    kernel = [-1, -1, -1, -1, 9, -1, -1, -1, -1]
    kernel = np.reshape(kernel, (3, 3))
    sharpened_image = ndimage.convolve(immagine, kernel)

    return np.copy(sharpened_image)


def IOU3D(stack_prediction, stack_label, classi):
    iou_classi = np.zeros(len(classi) + 1)
    percentuale_classi = np.zeros(len(classi))

    for t, value in enumerate(classi):
        print(f"Calcolo IOU per la classe {value}", end="\r")
        print('\n', end="\r")

        prediction_classe = np.where(stack_prediction == value, value, 0)
        label_classe = np.where(stack_label == value, value, 0)

        union_classe = np.where((prediction_classe + label_classe) > value, value, prediction_classe + label_classe)
        intersection_classe = np.where((prediction_classe * label_classe) > value, value, prediction_classe * label_classe)

        sum_union_classe = np.sum(union_classe == value)
        sum_intersection_classe = np.sum(intersection_classe == value)

        iou_classi[t] = sum_intersection_classe / sum_union_classe if sum_union_classe != 0 else 0
        percentuale_classi[t] = np.sum(stack_label == value)

    if len(percentuale_classi) > 1:
        totale_percentuale = np.sum(percentuale_classi)
        percentuale_classi /= totale_percentuale

        iou_classi[-1] = np.sum(iou_classi[:-1] * percentuale_classi)
        return iou_classi
    else:
        return iou_classi[0]

    return None


def resample(stack, step):
    """
    Resample a 3D numpy array along the third axis by a given step.
    
    Parameters:
    - stack: 3D numpy array to be resampled.
    - step: Integer step size for resampling.
    
    Returns:
    - new_stack: Resampled 3D numpy array.
    """
    # Calculate the new depth of the resampled stack
    new_depth = stack.shape[2] // step
    
    # Initialize the new stack with the correct shape and data type
    new_stack = np.empty((stack.shape[0], stack.shape[1], new_depth), dtype=stack.dtype)
    
    # Use array slicing to select every 'step' slice from the original stack
    new_stack = stack[:, :, :new_depth*step:step]
    
    return new_stack


def splitted_train(data_dir):
    folders = os.listdir(data_dir)
    folders.sort()
    if os.path.isfile(data_dir + folders[0]) or os.path.isfile(data_dir + folders[1]):
        return False
    elif (folders[0] == "images" or folders[0] == "images_train") and (
            folders[1] == "labels" or folders[1] == "labels_train"):
        return True


def splitted_test(data_dir):
    folders = os.listdir(data_dir)
    folders.sort()
    if os.path.isfile(data_dir + folders[0]) or os.path.isfile(data_dir + folders[1]):
        return False
    elif (folders[0] == "images" or folders[0] == "images_test") and (
            folders[1] == "labels" or folders[1] == "labels_test"):
        return True


def split_train(grayscale, label, dest_dir, name, estensione=".tif"):
    # Calcola le dimensioni del training e validation set
    lung_tot = 0
    i = 0
    while lung_tot != grayscale.shape[2]:
        len_val = int(grayscale.shape[2] * 25 / 100) - i
        len_train = int(grayscale.shape[2] - len_val)
        lung_tot = len_val + len_train
        i = i + 1

    # Genera indici casuali per la divisione
    indices = np.arange(grayscale.shape[2])
    train_indices, val_indices = train_test_split(indices, test_size=len_val, random_state=0)

    # Crea gli array per training e validation set
    grayscale_train = grayscale[:, :, train_indices]
    grayscale_val = grayscale[:, :, val_indices]
    label_train = label[:, :, train_indices]
    label_val = label[:, :, val_indices]

    # Salva le immagini e le etichette nei rispettivi directory
    for n in range(grayscale_train.shape[2]):
        nome = f"{name}{n:04d}"
        cv2.imwrite(f"{dest_dir}train/img/{nome}{estensione}", grayscale_train[:, :, n])
        cv2.imwrite(f"{dest_dir}train/label/label_{nome}{estensione}", label_train[:, :, n])

    for n in range(grayscale_val.shape[2]):
        nome = f"{name}{n:04d}"
        cv2.imwrite(f"{dest_dir}val/img/{nome}{estensione}", grayscale_val[:, :, n])
        cv2.imwrite(f"{dest_dir}val/label/label_{nome}{estensione}", label_val[:, :, n])


def determine_numpy_type(immagine):
    if immagine.format == 'TIFF':
        if immagine.mode == 'F':
            return np.float32
        elif immagine.mode == 'I':
            return np.uint32
        elif immagine.mode in {'I;16B', 'I;16', 'I;16L'}:
            return np.uint16
        elif immagine.mode in {'L', 'P'}:
            return np.uint8
    
    raise ValueError("Unsupported image format or mode")

def extract_resolution(immagine, step):
        exif = immagine.getexif()
        resolution = [1, 1, 1 * step]
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "XResolution":
                resolution[0] = float(value)
                resolution[2] = float(value) * step
            elif tag == "YResolution":
                resolution[1] = float(value)
        return resolution
    
def return_tointerpolate(grayscale, label):
    z = grayscale.shape[2]
    righe = grayscale.shape[0]
    colonne = grayscale.shape[1]
    indici = []
    for n in range(0, z, 1):
        conteggio_label = label[:, :, n].sum()
        if (conteggio_label == 0):
            indici.append(n)

    stack_grayscale_prevedere = np.empty([righe, colonne, len(indici)], dtype=grayscale.dtype)
    stack_label_prevedere = np.empty([righe, colonne, len(indici)], dtype=label.dtype)
    stack_grayscale_training = np.empty([righe, colonne, z - len(indici)], dtype=grayscale.dtype)
    stack_label_training = np.empty([righe, colonne, z - len(indici)], dtype=label.dtype)

    i = 0
    c = 0
    for n in range(0, z, 1):
        if (i < len(indici) and n == indici[i]):
            stack_grayscale_prevedere[:, :, i] = np.copy(grayscale[:, :, n])
            stack_label_prevedere[:, :, i] = np.copy(label[:, :, n])
            i = i + 1
        else:
            stack_grayscale_training[:, :, c] = np.copy(grayscale[:, :, n])
            stack_label_training[:, :, c] = np.copy(label[:, :, n])
            c = c + 1

    return stack_grayscale_training, stack_label_training, stack_grayscale_prevedere, stack_label_prevedere

def where_crop_ROI(array):
    z, rig, col = array.shape[2], array.shape[0], array.shape[1]

    lista_coordinate_X_min, lista_coordinate_Y_min, lista_coordinate_Z_min = [], [], []
    lista_coordinate_X_max, lista_coordinate_Y_max, lista_coordinate_Z_max = [], [], []

    for c in range(z):
        fetta = array[:, :, c]
        if fetta.mean() != 0:
            coords = np.argwhere(fetta != 0)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            lista_coordinate_X_min.append(x_min)
            lista_coordinate_Y_min.append(y_min)
            lista_coordinate_X_max.append(x_max)
            lista_coordinate_Y_max.append(y_max)

    for c in range(rig):
        fetta = array[c, :, :]
        if fetta.mean() != 0:
            coords = np.argwhere(fetta != 0)
            y_min, z_min = coords.min(axis=0)
            y_max, z_max = coords.max(axis=0)
            lista_coordinate_Z_min.append(z_min)
            lista_coordinate_Z_max.append(z_max)

    x_min, y_min, z_min = min(lista_coordinate_X_min), min(lista_coordinate_Y_min), min(lista_coordinate_Z_min)
    x_max, y_max, z_max = max(lista_coordinate_X_max), max(lista_coordinate_Y_max), max(lista_coordinate_Z_max)

    return [x_min, y_min, z_min, x_max, y_max, z_max]

def crop_stack(stack, coordinate):
    x_min, y_min, z_min, x_max, y_max, z_max = coordinate
    return stack[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]