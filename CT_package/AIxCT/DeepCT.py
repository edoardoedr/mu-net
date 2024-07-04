from ..Read_CT.CTDataset import CTDataset
from ..Read_CT.CTStack import CTStack
from .models.unet import UNet
from .models.segnet import SegNet
from .models.MedSAM import MedSAM
from .models.MedSAMLite import MedSAMLite
from .dataloaders import DataLoaderSegmentation_gray, DataLoaderSAM
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.multiprocessing import Pool, Process, set_start_method
import time
from functools import partial
import copy
import matplotlib.pyplot as plt
from .utils import *
from .pytorch_grad_cam import GradCAM
from monai.losses import DiceLoss
from tqdm import tqdm

keys_classe_deep = ["network", "tiles", "batch_size", "numero_classi", "retrain", "num_epochs", "use_box", "loss_function"]

class DeepCT:
    def __init__(self, dataset, parameters=None):
        # Controlla se il dataset caricato è una classe di tipo CT_dataset
        assert isinstance(dataset, CTDataset), "The dataset must be a dataset class"
        self.dataset = dataset

        # Inizializza i parametri
        self.parameters = self._initialize_parameters(parameters)

        # Imposta la directory di output
        self.output_dir = self._set_output_dir()

        # Imposta il dispositivo
        self.device = self._set_device()
        
    def dataloader(self):
        print("Initializing Datasets and Dataloaders...")
        data_dir = os.path.join(self.dataset.dataset_info["directory_principale"], "dataset_training")

        # Crea i dataset di training e validation
        if self.parameters["network"] == "MedSAM":
            image_datasets = {x: DataLoaderSAM(os.path.join(data_dir, x), 
                                               x, 
                                               self.parameters["tiles"], 
                                               use_box=self.parameters["use_box"],
                                               target_length= 1024) for x in ['train', 'val']}
        elif self.parameters["network"] == "MedSAMLite":
            image_datasets = {x: DataLoaderSAM(os.path.join(data_dir, x), 
                                               x, 
                                               self.parameters["tiles"], 
                                               use_box=self.parameters["use_box"],
                                               target_length= 226) for x in ['train', 'val']}
        else:
            image_datasets = {x: DataLoaderSegmentation_gray(os.path.join(data_dir, x), 
                                                             x, 
                                                             self.parameters["tiles"]) for x in ['train', 'val']}

        # Crea i dataloader di training e validation
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.parameters["batch_size"], shuffle=True, num_workers=0) for x in ['train', 'val']}

        return dataloaders_dict
    
    def train_dataset(self, subprocess = False):

        if subprocess == True:
            set_start_method('spawn', force=True)
            train = Process(target=self.main_training)
            train.start()
            train.join()
        else:
            self.main_training()
    
    def main_training(self):
        # Inizializza il modello
        n_channels = 1

        model = self._initialize_model(n_channels)

        # Carica i pesi pre-addestrati se specificato
        if self.parameters["retrain"]:
            state_dict = torch.load(self.parameters["retrain"], map_location=self.device)
            model.load_state_dict(state_dict)

        # Parallelizza il modello se necessario
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.to(self.device)

        # Log dei parametri da apprendere
        file_log = open(os.path.join(self.dataset.dataset_info["directory_output"], 'log_file.txt'), 'a')
        params_to_update = self._log_params_to_learn(model, file_log)
        file_log.close()

        # Ottimizzatore
        optimizer_ft = optim.Adam(params_to_update, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        # Funzione di perdita
        criterion = self._initialize_loss()

        # Addestramento e valutazione
        print("Train...")
        model_state_dict, hist = self.train_loop(model, optimizer_ft, criterion)

        # Salvataggio del modello
        print("Save...")
        self._save_model(model_state_dict)
        
    def train_loop(self, model, optimizer, criterion):
        since = time.time()
        best_model_state_dict = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        loss_h = []
        val_loss_h = []
        acc_h = []
        val_acc_h = []
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        dataloaders = self.dataloader()
        tqdm_iterabile = tqdm(range(1, self.parameters["num_epochs"] + 1), desc="training")

        for epoch in tqdm_iterabile:
            self.logger(f'Epoch {epoch}/{self.parameters["num_epochs"]}\n{"-" * 10}\n')

            for phase in ['train', 'val']:
                model.train() if phase == 'train' else model.eval()
                running_loss = 0.0
                running_iou_means = []

                for data in dataloaders[phase]:
                    
                    if "MedSAM" in self.parameters["network"]:
                        inputs, labels, boxes = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                    else:
                        inputs, labels = data[0].to(self.device), data[1].to(self.device)

                    if inputs.shape[0] == 1:
                        self.logger("Skipping iteration because batch_size = 1\n")
                        continue

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        if "MedSAM" in self.parameters["network"]:
                            outputs = model(inputs, boxes)
                        else:
                            outputs = model(inputs)
                        
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    iou_mean = iou2D(preds, labels, self.parameters["numero_classi"]).mean()
                    running_loss += loss.item() * inputs.size(0)
                    running_iou_means.append(iou_mean)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = np.array(running_iou_means).mean() if running_iou_means else 0.0

                self.logger(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

                if phase == 'train':
                    loss_h.append(epoch_loss)
                    acc_h.append(epoch_acc)
                    if epoch > 1:
                        ax[0].plot([epoch-1, epoch], loss_h[-2:], 'b')
                        ax[1].plot([epoch-1, epoch], acc_h[-2:], 'b')
                        ax[0].set_xlabel('Epochs', fontsize=14)
                        ax[0].set_ylabel('Loss', fontsize=14)
                        ax[1].set_xlabel('Epochs', fontsize=14)
                        ax[1].set_ylabel('Accuracy', fontsize=14)
                        plt.draw()

                elif phase == 'val':
                    val_loss_h.append(epoch_loss)
                    val_acc_h.append(epoch_acc)
                    if epoch > 1:
                        ax[0].plot([epoch-1, epoch], val_loss_h[-2:], 'r')
                        ax[1].plot([epoch-1, epoch], val_acc_h[-2:], 'r')
                        ax[0].set_xlabel('Epochs', fontsize=14)
                        ax[0].set_ylabel('Loss', fontsize=14)
                        ax[1].set_xlabel('Epochs', fontsize=14)
                        ax[1].set_ylabel('Accuracy', fontsize=14)
                        ax[0].legend(['Train_loss', 'Validation_loss'], fontsize=14)
                        ax[1].legend(['Train_accuracy', 'Validation_accuracy'], fontsize=14)
                        plt.savefig(os.path.join(self.output_dir, "epoch.png"))
                        plt.draw()

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_state_dict = copy.deepcopy(model.state_dict())
                    

        time_elapsed = time.time() - since
            
        self.logger(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
        self.logger(f'Best val Acc: {best_acc:.4f}\n')

        return best_model_state_dict, val_acc_h

    def inferenza(self, stack_topredict, axis):
        start_time = time.perf_counter()

        n_channels = 1
        model = self._initialize_model(n_channels, is_test = True)
        
        weight_dir = os.path.join(self.dataset.dataset_info["directory_output"], "pesi_modello.pth")
        state_dict = torch.load(weight_dir)
        model.load_state_dict(state_dict)

        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.to(self.device)
        model.eval()

        rig, col, zeta = stack_topredict.shape
        stack_inference = np.empty([rig, col, zeta], dtype=np.uint8)

        for x in range(stack_topredict.shape[axis]):
            if axis == "XY":
                imm_estratta = stack_topredict[:, :, x]
            elif axis == "XZ":
                imm_estratta = stack_topredict[x, :, :]
            elif axis == "YZ":
                imm_estratta = stack_topredict[:, x, :]
                
            if self.parameters["network"] == "MedSAM":
                image, boxes, padding = preprocess_SAM(imm_estratta, 1024)
                image, boxes = image.to(self.device), boxes.to(self.device)
                outputs = model(image, boxes)
                _, preds = torch.max(outputs, 1)
                preds = postprocess_SAM(preds, padding, imm_estratta)
                if axis == "XY":
                    stack_inference[:, :, x] = preds.squeeze(0).cpu().numpy().astype(np.uint8)
                elif axis == "XZ":
                    stack_inference[x, :, :] = preds.squeeze(0).cpu().numpy().astype(np.uint8)
                elif axis == "YZ":
                    stack_inference[:, x, :] = preds.squeeze(0).cpu().numpy().astype(np.uint8)
                
            else:
                image = unfold_image(imm_estratta, self.parameters["tiles"], patches=True).to(self.device)
                outputs = model(image)
                _, preds = torch.max(outputs, 1)
                preds = preds.to("cpu")
                if axis == "XY":
                    preds = fold_image(preds, rig, col, self.parameters["tiles"])
                    stack_inference[:, :, x] = preds.squeeze(0).cpu().numpy().astype(np.uint8)
                elif axis == "XZ":
                    preds = fold_image(preds, col, zeta, self.parameters["tiles"])
                    stack_inference[x, :, :] = preds.squeeze(0).cpu().numpy().astype(np.uint8)
                elif axis == "YZ":
                    preds = fold_image(preds, rig, zeta, self.parameters["tiles"])
                    stack_inference[:, x, :] = preds.squeeze(0).cpu().numpy().astype(np.uint8)

        del model
        finish_time = time.perf_counter()
        print(f"Immagini predette in {finish_time - start_time} seconds")
        torch.cuda.empty_cache() 
        torch.cuda.synchronize()

        return stack_inference

    def prediction(self, mode, subprocess=False):
        if subprocess:
            set_start_method('spawn', force=True)
            for grays in self.dataset.images:
                if mode == "2D":
                    self._predict_in_parallel(grays, ["XY"])
                elif mode == "3D":
                    self._predict_in_parallel(grays, ["XY", "XZ", "YZ"])
        else:
            for grays in self.dataset.images:
                if mode == "2D":
                    pred_result = self.inferenza(stack_topredict=grays.np_stack, axis="XY")
                    self._append_prediction(grays, pred_result)
                elif mode == "3D":
                    predizionis = [self.inferenza(stack_topredict=grays.np_stack, axis=asse) for asse in ["XY", "XZ", "YZ"]]
                    pred_tot = calcola_moda_parallelized(predizionis)
                    self._append_prediction(grays, pred_tot)

    def _predict_in_parallel(self, grays, axes):
        func_par = partial(self.inferenza, grays.np_stack)
        with Pool(processes=1) as pool:
            results = pool.map(func_par, axes)
        if len(axes) == 1:
            self._append_prediction(grays, results[0])
        else:
            pred_tot = calcola_moda_parallelized(results)
            self._append_prediction(grays, pred_tot)

    def _append_prediction(self, grays, pred_result):
        predizione = CTStack(np.copy(pred_result), grays.voxel_size, "prediction", grays.name)
        self.dataset.prediction.append(predizione)
        self.dataset.dataset_info["prediction"].append(f"{grays.name}_prediction")
    
    def _initialize_model(self, n_channels, is_test = False):
        
        if self.parameters["network"] == "UNET":
            return UNet(n_channels=n_channels, n_class=self.parameters["numero_classi"])
        
        elif "SEGNET" in self.parameters["network"]:
            if "VGG" in self.parameters["network"]:
                return SegNet(input_channels=n_channels, output_channels=self.parameters["numero_classi"], VGG=True)
            else:
                return SegNet(input_channels=n_channels, output_channels=self.parameters["numero_classi"], VGG=False)
        elif self.parameters["network"] == "MedSAM":
            use_boxes = False if is_test else self.parameters["use_boxes"]
            return MedSAM(self.parameters["numero_classi"], use_boxes)
        
        elif self.parameters["network"] == "MedSAMLite":
            use_boxes = False if is_test else self.parameters["use_boxes"]
            return MedSAMLite(self.parameters["numero_classi"], use_boxes)
        
    def _initialize_loss(self):
        
        if self.parameters["loss_function"] == "CE":
            weight = []
            return nn.CrossEntropyLoss(weight=(torch.FloatTensor(weight).to(self.device) if weight else None))
        
        elif self.parameters["loss_function"] == "DICE":
            return DiceLoss(include_background=False, to_onehot_y=True, sigmoid=False, softmax=False, squared_pred=False, jaccard=False, reduction= 'mean')
        
        elif self.parameters["loss_function"] == "IOU":
            return DiceLoss(include_background=False, to_onehot_y=True, sigmoid=False, softmax=False, squared_pred=False, jaccard=True, reduction= 'mean')
        
    def _log_params_to_learn(self, model, file_log):
        file_log.write("Params to learn:\n")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                file_log.write(f"{name}\n")
        return params_to_update

    def _save_model(self, model_state_dict):
        model_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
        layers_to_remove = [key for key in model_state_dict if "vgg16" in key]
        for key in layers_to_remove:
            del model_state_dict[key]
        torch.save(model_state_dict, os.path.join(self.output_dir, "pesi_modello.pth"))

    def _initialize_parameters(self, parameters):
        if isinstance(parameters, dict):
            params = {k: v for k, v in parameters.items() if k in keys_classe_deep}
        else:
            params = {k: self.dataset.dataset_info[k] for k in keys_classe_deep}
            
        if 'use_boxe' not in params.keys():
            params['use_boxe'] = False
            
        available_losses = ['CE', 'DICE', 'IOU']
            
        assert params["loss_function"] in available_losses, f"you have to choose a loss in {available_losses}"

        for key in keys_classe_deep:
            self.dataset.dataset_info[key] = params[key]

        return params

    def _set_output_dir(self):
        if self.dataset.dataset_info["modalità"] == "train":
            output_dir = return_output_dir(self.dataset.dataset_info["directory_principale"], self.dataset.dataset_info["nome"])
            self.dataset.dataset_info["directory_output"] = output_dir
        else:
            output_dir = self.dataset.dataset_info["directory_output"]
        return output_dir

    def _set_device(self):
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda")
            print("Parallelizzo su", torch.cuda.device_count(), "GPU")
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Uso una GPU")
        else:
            device = torch.device("cpu")
            print("Parallelizzo sul processore, uso", torch.get_num_threads(), "threads")
        return device
    
    def logger(self, message):
        
        with open(os.path.join(self.dataset.dataset_info["directory_output"], 'log_file.txt'), 'a') as file_log:
            file_log.write(message + '\n')
