import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import numpy as np
import cv2
import os
from PIL import Image, ImageFilter, ImageOps, ExifTags
import SimpleITK as sitk
import statistics as st
from scipy import ndimage, signal
import random
import time
from .utils import *
from functools import partial


############################ INIZIO CLASSE ##############################################################

class CTStack:
    def __init__(self, np_stack, voxel_size, tipology, name):
        """
        Inizializza un'istanza della classe CTStack.

        Parameters:
        np_stack (np.ndarray): L'array numpy che rappresenta lo stack.
        voxel_size (tuple or list): La dimensione del voxel come tupla o lista di tre elementi.
        tipology (str): La tipologia dello stack (deve essere una delle seguenti: 'image', 'label', 'prediction', 'explained').
        name (str): Il nome dello stack.
        """
        
        self.np_stack = self._validate_np_stack(np_stack)
        self.voxel_size =self. _validate_voxel_size(voxel_size)
        self.tipology = self._validate_tipology(tipology)
        self.name = self._validate_name(name)

    @staticmethod
    def read_stack(data_dir, step):
        if os.path.isdir(data_dir):
    
            lista_immagini = sorted(os.listdir(data_dir))
            data_dir = data_dir.rstrip("/")

            if not lista_immagini or not os.path.isfile(os.path.join(data_dir, lista_immagini[0])):
                print("La cartella selezionata non contiene solo immagini o non ne contiene.")
                return None

            immagine_prova = Image.open(os.path.join(data_dir, lista_immagini[0]))
            numpy_type = determine_numpy_type(immagine_prova)

            tipology = "label" if "label" in data_dir else "image"
            resolution = extract_resolution(immagine_prova, step)
            nome = os.path.basename(data_dir).split(".")[0]

            righe, colonne = np.array(immagine_prova).shape
            z = len(lista_immagini)
            zeta = z // step
            stack = np.empty((righe, colonne, zeta), dtype=numpy_type)

            for i, n in enumerate(range(0, z, step)):
                immagine = Image.open(os.path.join(data_dir, lista_immagini[n]))
                stack[:, :, i] = np.array(immagine)

            return CTStack(stack, resolution, tipology, nome)
        
        elif os.path.isfile(data_dir):
            stack = sitk.ReadImage(data_dir)
            stack_np = sitk.GetArrayFromImage(stack)
            stack_np = np.transpose(stack_np, (1, 2, 0))
            stack_np = resample(stack_np, step)
            resolution_in = stack.GetSpacing()
            resolution = [resolution_in[1], resolution_in[2], resolution_in[0] * step]
            name = os.path.basename(data_dir).split(".")[0]
            tipology = "label" if "label" in data_dir else "image"
            return CTStack(stack_np, resolution, tipology, name)

        else:
            raise ValueError("Invalid data directory or file path")
        
    def print_info(self):
        print(f"Le dimensioni dell'immagine sono: {self.np_stack.shape}")
        print(f"Il voxel ha dimensioni: {self.voxel_size}")
        print(f"La tipologia dello stack è: {self.tipology}")
        print(f"Il nome dello stack è: {self.name}")
        
        
    def resize_stack(self, new_size=None, scale=None):
        """
        Resize the stack to a new size or scale it by a factor.
        
        Parameters:
        - new_size: Tuple of 3 integers representing the new size.
        - scale: Tuple of 3 floats representing the scaling factor.
        """
        stack_sitk = sitk.GetImageFromArray(np.transpose(self.np_stack, (2, 0, 1)))
        original_size = stack_sitk.GetSize()
        original_spacing = stack_sitk.GetSpacing()

        if new_size:
            assert len(new_size) == 3, "new_size must be a tuple of 3 elements"
            new_size = [ns if ns != "o" else osz for ns, osz in zip(new_size, original_size)]
        elif scale:
            assert len(scale) == 3, "scale must be a tuple of 3 elements"
            new_size = [int(osz * s) for osz, s in zip(original_size, scale)]
        else:
            raise ValueError("Either new_size or scale must be provided")

        new_spacing = [os * osz / ns for os, osz, ns in zip(original_spacing, original_size, new_size)]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(stack_sitk.GetOrigin())
        resampler.SetOutputDirection(stack_sitk.GetDirection())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if self.typology in ["label", "prediction"] else sitk.sitkLinear)

        resized_stack_sitk = resampler.Execute(stack_sitk)
        resized_stack_np = sitk.GetArrayFromImage(resized_stack_sitk)
        resized_stack_np = np.transpose(resized_stack_np, (1, 2, 0))

        return CTStack(resized_stack_np, new_spacing, self.typology, self.name)


    def save_stack(self, dest_dir, extension):

        # Salva lo stack come file immagine
        if extension in [".tif",".png"]:
            directory = os.path.join(dest_dir, self.name)
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            for n in range(self.np_stack.shape[2]):
                file_name = f"{self.name}/{self.name}{n:04d}{extension}"
                cv2.imwrite(os.path.join(dest_dir, file_name), self.np_stack[:, :, n])
            
            print(f"Stack saved as a sequence of {extension}")
        
        else:
            # Trasponi e converti lo stack per altre estensioni
            stack_tosave = np.transpose(self.np_stack, (2, 0, 1))
            stack_sitk = sitk.GetImageFromArray(stack_tosave)
            stack_sitk.SetSpacing([self.voxel_size[2], self.voxel_size[0], self.voxel_size[1]])

            # Determina il nome del file e la tipologia
            typology = ""
            if self.typology == "label" and "label" not in self.name:
                typology = "_label"
            elif self.typology == "prediction":
                typology = "_prediction"
            elif self.typology == "explained":
                typology = "_explained"

            # Rimuovi "_label" dal nome se presente
            name = self.name.replace("_label", "") if "label" in self.name else self.name
            name = self.name.replace("_image", "") if "image" in self.name else self.name

            # Salva lo stack con la tipologia e l'estensione appropriate
            sitk.WriteImage(stack_sitk, os.path.join(dest_dir, f"{name}{typology}{extension}"))

            print(f"Stack saved as {extension}")

    def crop_ROI(self, coordinate=[]):

        if len(coordinate) > 1:
            self.np_stack = crop_stack(self.np_stack, coordinate)
        else:
            coordinate = where_crop_ROI(self.np_stack)
            self.np_stack = crop_stack(self.np_stack, coordinate)

    @staticmethod
    def multiply(stack_one, stack_two):
        assert isinstance(stack_one, CTStack), "Le immagini devono essere di tipo stack images"
        assert isinstance(stack_two, CTStack), "Le immagini devono essere di tipo stack images"

        new_stack_array = stack_one.np_stack * stack_two.np_stack
        nuovo_nome = stack_one.name
        nuovo_voxel_size = stack_one.voxel_size
        nuovo_typology = stack_one.tipology

        return CTStack(new_stack_array, nuovo_voxel_size, nuovo_typology, nuovo_nome)

    def copy_stack(self):
        new_stack = np.copy(self.np_stack)

        return CTStack(new_stack, self.voxel_size, self.tipology, self.name)

    def mode_filter_25D(self, filter_size):
        start_time = time.perf_counter()
        rig, col, zeta = self.np_stack.shape

        filtro = partial(filtro_mode, filter_size)

        # Filtriamo su xy
        list_stack = [self.np_stack[:, :, n] for n in range(zeta)]
        with Pool() as pool:
            result = pool.map(filtro, list_stack)
        for n in range(zeta):
            self.np_stack[:, :, n] = result[n]

        # Filtriamo su xz
        list_stack = [self.np_stack[n, :, :] for n in range(rig)]
        with Pool() as pool:
            result = pool.map(filtro, list_stack)
        for n in range(rig):
            self.np_stack[n, :, :] = result[n]

        finish_time = time.perf_counter()
        print(f"Filtro mode finito in {finish_time - start_time:.2f} seconds - using multiprocessing")
        print("---")

        return CTStack(self.np_stack, self.voxel_size, self.tipology, self.name)
    
    def median_filter_25D(self, filter_size):
        start_time = time.perf_counter()
        rig, col, zeta = self.np_stack.shape

        filtro = partial(signal.medfilt2d, kernel_size=filter_size)

        # Filtriamo su xy
        list_stack = [self.np_stack[:, :, n] for n in range(zeta)]
        with Pool() as pool:
            result = pool.map(filtro, list_stack)
        for n in range(zeta):
            self.np_stack[:, :, n] = result[n]

        # Filtriamo su xz
        list_stack = [self.np_stack[n, :, :] for n in range(rig)]
        with Pool() as pool:
            result = pool.map(filtro, list_stack)
        for n in range(rig):
            self.np_stack[n, :, :] = result[n]

        finish_time = time.perf_counter()
        print(f"Filtro median finito in {finish_time - start_time:.2f} seconds - using multiprocessing")
        print("---")

        return CTStack(self.np_stack, self.voxel_size, self.tipology, self.name)
    
    def unsharpmask(self):
        start_time = time.perf_counter()
        rig, col, zeta = self.np_stack.shape

        filtro = FiltroUnsharp

        # Filtriamo su xy
        list_stack = [self.np_stack[:, :, n] for n in range(zeta)]
        with Pool() as pool:
            result = pool.map(filtro, list_stack)
        for n in range(zeta):
            self.np_stack[:, :, n] = result[n]

        if self.np_stack.dtype == np.float32:
            massimo = self.np_stack.max()
            minimo = self.np_stack.min()
            self.np_stack = (self.np_stack - minimo) / (massimo - minimo)

        finish_time = time.perf_counter()
        print(f"Filtro unsharpmask finito in {finish_time - start_time:.2f} seconds - using multiprocessing")
        print("---")

        return CTStack(self.np_stack, self.voxel_size, self.tipology, self.name)
    
    def fill_holes(self, label_riempire):
        start_time = time.perf_counter()
        rig, col, zeta = self.np_stack.shape
        struttura = np.ones((5, 5))

        filtro = partial(fill_holed2D, label_riempire, struttura)

        # Filtriamo su xy
        list_stack = [self.np_stack[:, :, n] for n in range(zeta)]
        with Pool() as pool:
            result = pool.map(filtro, list_stack)
        for n in range(zeta):
            self.np_stack[:, :, n] = result[n]

        # Filtriamo su xz
        list_stack = [self.np_stack[n, :, :] for n in range(rig)]
        with Pool() as pool:
            result = pool.map(filtro, list_stack)
        for n in range(rig):
            self.np_stack[n, :, :] = result[n]

        finish_time = time.perf_counter()
        print(f"Filtro fill holes finito in {finish_time - start_time:.2f} seconds - using multiprocessing")
        print("---")
        
        return CTStack(self.np_stack, self.voxel_size, self.tipology, self.name)
    
    def sharpen(self):
        start_time = time.perf_counter()
        rig, col, zeta = self.np_stack.shape

        filtro = Filtrosharpen

        # Filtriamo su xy
        list_stack = [self.np_stack[:, :, n] for n in range(zeta)]
        with Pool() as pool:
            result = pool.map(filtro, list_stack)
        for n in range(zeta):
            self.np_stack[:, :, n] = result[n]

        if self.np_stack.dtype == np.float32:
            minimo = self.np_stack.min()
            self.np_stack[self.np_stack == 0] = minimo
            massimo = self.np_stack.max()
            self.np_stack = (self.np_stack - minimo) / (massimo - minimo)

        finish_time = time.perf_counter()
        print(f"Filtro sharpen finito in {finish_time - start_time:.2f} seconds - using multiprocessing")
        print("---")
        
        return CTStack(self.np_stack, self.voxel_size, self.tipology, self.name)

    def change_values(self, valori, substitute):
        if len(valori) == 2 and valori[1] == "=":
            self.np_stack[self.np_stack == valori[0]] = substitute
        elif len(valori) == 2 and valori[1] == ">":
            self.np_stack[self.np_stack > valori[0]] = substitute
        elif len(valori) == 2 and valori[1] == "<":
            self.np_stack[self.np_stack < valori[0]] = substitute
        elif len(valori) == 3 and valori[1] == "<>":
            self.np_stack[(self.np_stack < valori[0]) | (self.np_stack > valori[2])] = substitute
        elif len(valori) == 3 and valori[1] == "><":
            self.np_stack[(self.np_stack > valori[0]) & (self.np_stack < valori[2])] = substitute
        return CTStack(self.np_stack, self.voxel_size, self.tipology, self.name)
    
    def convert(self, stack_type):
        if stack_type == "8bit":
            stack_8bit = self.np_stack.astype(np.float32)
            stack_8bit = np.clip(stack_8bit, 0, None)
            stack_8bit = (stack_8bit / stack_8bit.max()) * 255
            stack_8bit = np.round(stack_8bit).astype(np.uint8)
            return CTStack(stack_8bit, self.voxel_size, self.tipology, self.name)
        elif stack_type == "16bit":
            stack_16bit = self.np_stack.astype(np.float32)
            stack_16bit = np.clip(stack_16bit, 0, None)
            stack_16bit = (stack_16bit / stack_16bit.max()) * 255
            stack_16bit = np.round(stack_16bit).astype(np.uint16)
            return CTStack(stack_16bit, self.voxel_size, self.tipology, self.name)
        elif stack_type == "normalize":
            stack_01 = self.np_stack.astype(np.float32)
            stack_01 = np.clip(stack_01, 0, None)
            stack_01 = stack_01 / stack_01.max()
            return CTStack(stack_01, self.voxel_size, self.tipology, self.name)
        
    def _validate_np_stack(self, np_stack):
            if not isinstance(np_stack, np.ndarray):
                raise TypeError("Array must be a numpy array")
            return np_stack

    def _validate_voxel_size(self, voxel_size):
            if not (isinstance(voxel_size, (tuple, list)) and len(voxel_size) == 3):
                raise TypeError("Voxel size must be a tuple or a list of three elements")
            return voxel_size

    def _validate_tipology(self, tipology):
            valid_tipologies = {"image", "label", "prediction", "explained"}
            if not (isinstance(tipology, str) and tipology in valid_tipologies):
                raise TypeError("Tipology must be one of the following: 'image', 'label', 'prediction', 'explained'")
            return tipology

    def _validate_name(self, name):
            if not isinstance(name, str):
                raise TypeError("Name must be a string")
            return name