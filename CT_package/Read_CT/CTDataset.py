import numpy as np
from .CTStack import CTStack
from .utils import *
import os
import copy
import json
import pprint


keys_classe = [
    "directory_principale", "nome", "images", "labels", "directory_output", 
    "numero_classi", "lista_filtri_grays", "lista_filtri_labels", "modalità", 
    "prediction", "intersection_over_union", "explained"
]

class CTDataset:
    def __init__(self, data_dir=None, step=None, images=None, labels=None):
        self.dataset_info = {key: [] for key in keys_classe}
        self.dataset_info["explained"] = "No"
        self.images = images if images is not None else []
        self.labels = labels if labels is not None else []
        self.prediction = []
        self.grad_CAM = []

        if data_dir:
            self._initialize_from_directory(data_dir, step)
        else:
            self._validate_stacks()

    @staticmethod
    def merge(dataset_uno, *args):
        """
        Unisce uno o più dataset a un dataset principale.
        
        :param dataset_uno: Il dataset principale.
        :param args: Uno o più dataset da unire al dataset principale.
        :return: Un nuovo dataset contenente i dati combinati.
        """
        lista_images = dataset_uno.images
        lista_labels = dataset_uno.labels
        lista_prediction = dataset_uno.prediction
        lista_grad_CAM = dataset_uno.grad_CAM

        images_list = dataset_uno.dataset_info["images"]
        labels_list = dataset_uno.dataset_info["labels"]
        lista_filtri_grays = dataset_uno.dataset_info["lista_filtri_grays"]
        lista_filtri_labels = dataset_uno.dataset_info["lista_filtri_labels"]
        prediction_list = dataset_uno.dataset_info["prediction"]

        for datas in args:
            lista_images.extend(datas.images)
            lista_labels.extend(datas.labels)
            lista_prediction.extend(datas.prediction)
            lista_grad_CAM.extend(datas.grad_CAM)
            images_list.extend(datas.dataset_info["images"])
            labels_list.extend(datas.dataset_info["labels"])
            lista_filtri_grays.extend(datas.dataset_info["lista_filtri_grays"])
            lista_filtri_labels.extend(datas.dataset_info["lista_filtri_labels"])
            prediction_list.extend(datas.dataset_info["prediction"])

        new_dataset = CTDataset(images=lista_images, labels=lista_labels)
        new_dataset.prediction = lista_prediction
        new_dataset.grad_CAM = lista_grad_CAM
        new_dataset.dataset_info = copy.deepcopy(dataset_uno.dataset_info)
        new_dataset.dataset_info["images"] = images_list
        new_dataset.dataset_info["labels"] = labels_list
        new_dataset.dataset_info["lista_filtri_grays"] = lista_filtri_grays
        new_dataset.dataset_info["lista_filtri_labels"] = lista_filtri_labels
        new_dataset.dataset_info["prediction"] = prediction_list

        return new_dataset

    def reset_dataset(self):
        """
        Reimposta vari attributi del dataset, riportandolo a uno stato iniziale.
        """
        self.dataset_info["lista_filtri_grays"] = []
        self.dataset_info["lista_filtri_labels"] = []
        self.dataset_info["prediction"] = []
        self.dataset_info["intersection_over_union"] = []
        self.dataset_info["explained"] = "No"
        self.dataset_info["directory_output"] = []
        self.prediction = []
        self.grad_CAM = []


    def copy_dataset(self):
        """
        Crea una copia profonda del dataset corrente, inclusi immagini, etichette, predizioni e grad_CAM.
        """
        new_lista_images = [img.copy_stack() for img in self.images]
        new_lista_labels = [lbl.copy_stack() for lbl in self.labels]
        new_lista_prediction = [pred.copy_stack() for pred in self.prediction]
        new_lista_grad_CAM = [grad.copy_stack() for grad in self.grad_CAM]

        new_dataset = CTDataset(images=new_lista_images, labels=new_lista_labels)
        new_dataset.dataset_info = copy.deepcopy(self.dataset_info)
        new_dataset.prediction = new_lista_prediction
        new_dataset.grad_CAM = new_lista_grad_CAM

        return new_dataset

    def change_values(self, valori, substitute, tipologia="label"):
        """
        Cambia i valori specificati nelle immagini, etichette o predizioni del dataset.
        
        :param valori: Valori da cambiare.
        :param substitute: Valori sostitutivi.
        :param tipologia: Tipo di dati da modificare ("label", "image", "prediction").
        """
        if tipologia == "label":
            self.labels = [dati.change_values(valori, substitute) for dati in self.labels]
        elif tipologia == "image":
            self.images = [dati.change_values(valori, substitute) for dati in self.images]
        elif tipologia == "prediction":
            self.prediction = [dati.change_values(valori, substitute) for dati in self.prediction]

    def label_x_images(self):
        """
        Moltiplica ogni immagine per la corrispondente etichetta.
        """
        self.images = [CTStack.multiply(img, lbl) for img, lbl in zip(self.images, self.labels)]

    def save_dataset(self):
        """
        Salva il dataset nelle directory specificate.
        """
        dest_dir = os.path.join(self.dataset_info["directory_principale"], self.dataset_info["nome"])
        os.makedirs(dest_dir, exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "labels"), exist_ok=True)

        for img, labl in zip(self.images, self.labels):
            img.save_stack(dest_dir=os.path.join(dest_dir, "images"), estenzione=".nii.gz")
            labl.save_stack(dest_dir=os.path.join(dest_dir, "labels"), estenzione=".nii.gz")

        print("Dataset saved")

    def crop_ROI_dataset(self):
        """
        Ritaglia la regione di interesse (ROI) per ogni immagine ed etichetta nel dataset.
        """
        new_images = []
        new_labels = []
        for image, label in zip(self.images, self.labels):
            coordinate = where_crop_ROI(label.np_stack)
            new_images.append(image.crop_ROI(coordinate=coordinate))
            new_labels.append(label.crop_ROI(coordinate=coordinate))

        self.images = new_images
        self.labels = new_labels

    def resize_dataset(self, new_size=None, scale=None):
        """
        Ridimensiona le immagini e le etichette del dataset.
        
        :param new_size: Nuova dimensione per il ridimensionamento.
        :param scale: Scala per il ridimensionamento.
        """
        self.images = [image.resize_stack(new_size, scale) for image in self.images]
        self.labels = [label.resize_stack(new_size, scale) for label in self.labels]
        

    def mode_filter_25D(self, filter_size, tipologia="label"):
        """
        Applica un filtro di moda 2.5D alle immagini, etichette o predizioni del dataset.
        
        :param filter_size: Dimensione del filtro.
        :param tipologia: Tipo di dati da filtrare ("label", "image", "prediction").
        """
        if tipologia == "label":
            self.labels = [dati.mode_filter_25D(filter_size) for dati in self.labels]
            self.dataset_info["lista_filtri_labels"].append("mode")
        elif tipologia == "image":
            self.images = [dati.mode_filter_25D(filter_size) for dati in self.images]
            self.dataset_info["lista_filtri_grays"].append("mode")
        elif tipologia == "prediction":
            self.prediction = [dati.mode_filter_25D(filter_size) for dati in self.prediction]

    def median_filter_25D(self, filter_size, tipologia="label"):
        """
        Applica un filtro mediano 2.5D alle immagini, etichette o predizioni del dataset.
        
        :param filter_size: Dimensione del filtro.
        :param tipologia: Tipo di dati da filtrare ("label", "image", "prediction").
        """
        if tipologia == "label":
            self.labels = [dati.median_filter_25D(filter_size) for dati in self.labels]
            self.dataset_info["lista_filtri_labels"].append("median")
        elif tipologia == "image":
            self.images = [dati.median_filter_25D(filter_size) for dati in self.images]
            self.dataset_info["lista_filtri_grays"].append("median")
        elif tipologia == "prediction":
            self.prediction = [dati.median_filter_25D(filter_size) for dati in self.prediction]

    def unsharpmask(self):
        """
        Applica un filtro di maschera di contrasto (unsharp mask) alle immagini del dataset.
        """
        self.images = [dati.unsharpmask() for dati in self.images]
        self.dataset_info["lista_filtri_grays"].append("unsharp")

    def convert(self, stack_type):
        """
        Converte le immagini del dataset nel tipo di stack specificato.
        
        :param stack_type: Tipo di stack a cui convertire le immagini.
        """
        self.images = [dati.convert(stack_type) for dati in self.images]

    def sharpen(self):
        """
        Applica un filtro di nitidezza (sharpen) alle immagini del dataset.
        """
        self.images = [dati.sharpen() for dati in self.images]
        self.dataset_info["lista_filtri_grays"].append("sharpen")

    def fill_holes(self, label_riempire, tipologia="prediction"):
        """
        Riempie i buchi nelle etichette o nelle predizioni del dataset.
        
        :param label_riempire: Lista di etichette da riempire.
        :param tipologia: Tipo di dati da riempire ("label" o "prediction").
        """
        if tipologia == "label":
            self.labels = [dati.fill_holes(label_riempire) for dati in self.labels]
            self.dataset_info["lista_filtri_labels"].append("fill_holes")
        elif tipologia == "prediction":
            self.prediction = [dati.fill_holes(label_riempire) for dati in self.prediction]

    def delete_stack(self, name):
        """
        Elimina le immagini e le etichette con il nome specificato dal dataset.
        
        :param name: Nome dello stack da eliminare.
        """
        indices_to_remove = [i for i, image in enumerate(self.images) if image.name == name]
        for index in sorted(indices_to_remove, reverse=True):
            self.images.pop(index)
            self.labels.pop(index)

    def inizializza_test(self, cls_or_dir):
        """
        Inizializza il dataset per il test, copiando le informazioni da un altro dataset o caricandole da un file.
        
        :param cls_or_dir: Un'istanza di CTDataset o un percorso di file contenente le informazioni del dataset.
        """
        old_dict = copy.deepcopy(self.dataset_info)

        if isinstance(cls_or_dir, CTDataset):
            self.dataset_info = copy.deepcopy(cls_or_dir.dataset_info)
        elif isinstance(cls_or_dir, str):
            with open(cls_or_dir, "r") as file:
                self.dataset_info = json.load(file)
            self.dataset_info["directory_output"] = os.path.dirname(cls_or_dir) + "/"

        if not self.labels:
            self.dataset_info["modalità"] = "test_predici"
        else:
            assert all(isinstance(x, CTStack) for x in self.labels), "Labels must be a stack of CTStack instances in test_performance dataset"
            assert len(self.images) == len(self.labels), "Labels must have the same length as images in test_performance dataset"
            self.dataset_info["modalità"] = "test_performance"

        self.dataset_info.update({
            "images": old_dict["images"],
            "labels": old_dict["labels"],
            "directory_principale": old_dict["directory_principale"],
            "nome": old_dict["nome"],
            "prediction": old_dict["prediction"],
            "intersection_over_union": []
        })

    def performance(self):
        """
        Calcola la performance del modello in termini di Intersection over Union (IoU) per il dataset di test.
        """
        if self.dataset_info["modalità"] == "test_performance":
            ious = []
            for lab, pred in zip(self.labels, self.prediction):
                iou = IOU3D(pred.np_stack, lab.np_stack, list(range(1, self.dataset_info["numero_classi"])))
                print(iou)
                ious.append(iou)
            self.dataset_info["intersection_over_union"].append(ious)

    def save_prediction(self):
        """
        Salva le predizioni del dataset nella directory di output.
        """
        dest_dir = os.path.dirname(self.dataset_info["directory_output"]) + "/"
        for preds in self.prediction:
            preds.save_stack(dest_dir, ".nii.gz")

    def save_grad_CAM(self):
        """
        Salva le mappe di attivazione Grad-CAM del dataset nella directory di output.
        """
        dest_dir = os.path.dirname(self.dataset_info["directory_output"]) + "/"
        for preds in self.grad_CAM:
            preds.save_stack(dest_dir, ".nii.gz")

    def print_info(self):
        """
        Stampa le informazioni del dataset.
        """
        pprint.pprint(self.dataset_info)

    def save_info(self):
        """
        Salva le informazioni del dataset in un file JSON nella directory di output.
        """
        with open(os.path.join(self.dataset_info["directory_output"], self.dataset_info["nome"] + ".json"), "w") as file:
            json.dump(self.dataset_info, file)

    def split_dataset(self):
        """
        Divide il dataset in set di addestramento e validazione e salva i dati nelle rispettive directory.
        """
        dest_dir = os.path.join(self.dataset_info["directory_principale"], "dataset_training/")
        os.makedirs(dest_dir, exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "train/img"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "train/label"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "val/img"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "val/label"), exist_ok=True)

        # Rimuove i file esistenti nelle directory di addestramento e validazione
        for subdir in ["train/img", "train/label", "val/img", "val/label"]:
            file_list = os.listdir(os.path.join(dest_dir, subdir))
            if file_list:
                for file in file_list:
                    os.remove(os.path.join(dest_dir, subdir, file))

        # Divide e salva le immagini e le etichette
        for images, labels in zip(self.images, self.labels):
            split_train(images.np_stack, labels.np_stack, dest_dir, images.name)


    def _initialize_from_directory(self, data_dir, step):
        dataset = self.upload_dataset(data_dir, step)
        self.images, self.labels = dataset
        base_dir = os.path.dirname(data_dir.rstrip("/")) + "/"
        self.dataset_info.update({
            "directory_principale": base_dir,
            "directory_output": base_dir,
            "nome": os.path.basename(data_dir.rstrip("/"))
        })
        self.dataset_info["images"] = [img.name for img in self.images]
        self.dataset_info["labels"] = [lbl.name for lbl in self.labels]
        self._set_modalità()

    def _validate_stacks(self):
        assert all(isinstance(x, CTStack) for x in self.images), "Images must be a stack of CTStack instances"
        assert all(isinstance(x, CTStack) for x in self.labels), "Labels must be a stack of CTStack instances"
        assert all(isinstance(x, CTStack) for x in self.prediction), "Predictions must be a stack of CTStack instances"
        assert all(isinstance(x, CTStack) for x in self.grad_CAM), "Grad_CAM must be a stack of CTStack instances"

    def _set_modalità(self):
        nome = self.dataset_info["nome"]
        if "train" in nome.lower():
            self.dataset_info["modalità"] = "train"
            assert len(self.images) == len(self.labels), "Labels must have the same length as images in train dataset"
        elif "test" in nome.lower():
            self.dataset_info["modalità"] = "test_predici" if not self.labels else "test_performance"
            if self.labels:
                assert len(self.images) == len(self.labels), "Labels must have the same length as images in test_performance dataset"

    @staticmethod
    def upload_dataset(data_dir, step):
        files_or_folders = sorted(os.listdir(data_dir))
        if len(files_or_folders) == 2 and (splitted_train(data_dir) or splitted_test(data_dir)):
            return CTDataset._process_split_dataset(data_dir, files_or_folders, step)
        else:
            return CTDataset._process_single_folder(data_dir, files_or_folders, step)

    @staticmethod
    def _process_split_dataset(data_dir, folders, step):
        images_dir, labels_dir = [os.path.join(data_dir, folder) for folder in folders]
        image_files = sorted(os.listdir(images_dir))
        label_files = sorted(os.listdir(labels_dir))
        if len(image_files) != len(label_files):
            raise TypeError("Mismatch between images and labels count")

        images = [CTStack.read_stack(os.path.join(images_dir, img), step) for img in image_files]
        labels = [CTStack.read_stack(os.path.join(labels_dir, lbl), step) for lbl in label_files]
        return images, labels

    @staticmethod
    def _process_single_folder(data_dir, files, step):
        image_files = [os.path.join(data_dir, f) for f in files if "label" not in f]
        label_files = [os.path.join(data_dir, f) for f in files if "label" in f]

        images = [CTStack.read_stack(img, step) for img in image_files]
        labels = [CTStack.read_stack(lbl, step) for lbl in label_files] if label_files else []
        return images, labels
