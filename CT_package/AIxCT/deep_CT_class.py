from ..Read_CT.CT_dataset import CT_dataset
from ..Read_CT.CT_stack_class import CT_stack
from .models.unet import UNet
from .models.segnet import SegNet
from .models.continual_Segnet import Continual_SegNet
from .dataloaders import DataLoaderSegmentation_gray, DataLoaderSegmentation_rgb
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.multiprocessing import Pool, cpu_count, Process, set_start_method
import time
from functools import partial
import copy
import matplotlib.pyplot as plt
from .utils import *
from PIL import Image, ImageFilter, ImageOps, ExifTags
import statistics as st
from .pytorch_grad_cam import GradCAM
from .pytorch_grad_cam.utils.image import show_cam_on_image
import pprint
import json
from tqdm import tqdm

keys_classe_deep = ["network", "tiles", "batch_size", "numero_classi", "retrain", "num_epochs"]

class deep_CT:

    def __init__(self, dataset, parameters = None, keep_feature_extract = None):
        # controlla se il dataset caricato è una classe di tipo CT dataset
        assert isinstance(dataset, CT_dataset), "The dataset must be a dataset class"
        self.dataset = dataset

        if isinstance(parameters, dict):
            self.parameters = {k: v for k, v in parameters.items() if k in keys_classe_deep}
            self.dataset.dataset_info["network"] = self.parameters["network"]
            self.dataset.dataset_info["tiles"] = self.parameters["tiles"]
            self.dataset.dataset_info["batch_size"] = self.parameters["batch_size"]
            self.dataset.dataset_info["numero_classi"] = self.parameters["numero_classi"]
            self.dataset.dataset_info["retrain"] = self.parameters["retrain"]
            self.dataset.dataset_info["num_epochs"] = self.parameters["num_epochs"]
        else:
            self.parameters = {}
            self.parameters["network"] = self.dataset.dataset_info["network"]
            self.parameters["tiles"] = self.dataset.dataset_info["tiles"]
            self.parameters["batch_size"] = self.dataset.dataset_info["batch_size"]
            self.parameters["numero_classi"] = self.dataset.dataset_info["numero_classi"]
            self.parameters["retrain"] = self.dataset.dataset_info["retrain"]
            self.parameters["num_epochs"] = self.dataset.dataset_info["num_epochs"]

        self.parameters["data_type"] = "gray"
        self.keep_feature_extract = True
        if self.dataset.dataset_info["modalità"] == "train":
            print("sono nel train, creo la dir")
            self.output_dir = return_output_dir(self.dataset.dataset_info["directory_principale"], self.dataset.dataset_info["nome"] )
            self.dataset.dataset_info["directory_output"] = self.output_dir
        else:
            self.output_dir = self.dataset.dataset_info["directory_output"]

        if torch.cuda.device_count() > 1:
            self.device = torch.device("cuda")
            print("parallelizzo su ", torch.cuda.device_count()," GPU")
        elif torch.cuda.is_available():    
            self.device = torch.device("cuda:0")
            print("uso una GPU")
        else:
            self.device = torch.device("cpu")
            #se si vuol limitare il numero di threads usa questo
            #torch.set_num_threads(4)
            print("parallelizzo sul processore, uso ", torch.get_num_threads(), "threads")
            

    def dataloader(self):
        
        print("Initializing Datasets and Dataloaders...")
        data_dir = self.dataset.dataset_info["directory_principale"] + "dataset_training/"
         # Create training and validation datasets
        if self.parameters["data_type"] == "gray":
            image_datasets = {x: DataLoaderSegmentation_gray(os.path.join(data_dir, x), x, self.parameters["tiles"]) for x in ['train', 'val']}
        else:
            image_datasets = {x: DataLoaderSegmentation_rgb(os.path.join(data_dir, x), x, self.parameters["tiles"]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = self.parameters["batch_size"], shuffle=True, num_workers=0) for x in ['train', 'val']}
        
        return dataloaders_dict

    def main_training(self):
        # Initialize model --- QUI PUOI CAMBIARE IL TIPO DI RETE DA USARE
        #model_deeplabv3, input_size = initialize_model(num_classes, keep_feature_extract, use_pretrained=True) #deeplabv3
        if self.parameters["data_type"] == "RGB":
            n_channels = 3
        else:
            n_channels = 1
        
        if self.parameters["network"] == "UNET":
            model = UNet(n_channels = n_channels, n_class = self.parameters["numero_classi"])

        elif "SEGNET" in self.parameters["network"]:
            if "VGG" in self.parameters["network"]:
                model = SegNet(input_channels = n_channels, output_channels = self.parameters["numero_classi"], VGG = True) #segnet
            elif "Continual" in self.parameters["network"]:
                model = Continual_SegNet(input_channels = n_channels, output_channels = self.parameters["numero_classi"])
            else:
                model = SegNet(input_channels = n_channels, output_channels = self.parameters["numero_classi"], VGG = False) #segnet

        if len(self.parameters["retrain"])>1:
            state_dict = torch.load(self.parameters["retrain"], map_location = self.device)
            model.load_state_dict(state_dict)

        if self.device == torch.device("cuda"):
            model = nn.DataParallel(model, device_ids=[0,1])
        
        model = model.to(self.device)

        file_log = open(self.dataset.dataset_info["directory_output"] + 'log_file.txt', 'a')

        params_to_update = model.parameters()
        #print("Params to learn:")
        file_log.write("Params to learn:")
        if self.keep_feature_extract:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    file_log.write(f"\n {name}")
                    #print("\t", name)
        else:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    file_log.write(f"\n {name}")
                    #print("\t", name)
        file_log.write("\n")

        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(params_to_update, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) #
        #optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        weight = []
        # Setup the loss function
        criterion = nn.CrossEntropyLoss(weight=(torch.FloatTensor(weight).to(self.device) if weight else None))

        # Prepare output directory
        #pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        print("Train...")
        # Train and evaluate
        file_log.close()
        model_state_dict, hist = self.train_loop(model, optimizer_ft, criterion)

        print("Save ...")
        model_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
        layers_to_remove = []
        for key, value in model_state_dict.items():
            if "vgg16" in key:
                layers_to_remove.append(key)
        for key in layers_to_remove:
            del model_state_dict[key]

        torch.save(model_state_dict, os.path.join(self.output_dir, "pesi_modello.pth"))


    def train_loop(self, model, optimizer, criterion):

        since = time.time()
        val_acc_history = []
        best_model_state_dict = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        counter = 0
        loss_h = []
        val_loss_h = []
        acc_h = []
        val_acc_h = []
        fig, ax = plt.subplots(1, 2, figsize=(10,10))
        dataloaders = self.dataloader()
        iterabile = range(1, self.parameters["num_epochs"] + 1)
        tqdm_iterabile = tqdm(iterabile, desc="training")


        for epoch in tqdm_iterabile:
            file_log = open(self.dataset.dataset_info["directory_output"] + 'log_file.txt', 'a')
            file_log.write('Epoch {}/{}'.format(epoch, self.parameters["num_epochs"]))
            file_log.write('-' * 10)
            #print('Epoch {}/{}'.format(epoch, self.parameters["num_epochs"]))
            #print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_iou_means = []

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs =  inputs.transpose( 1, 3) #to have BATCH;C;H;W
                    inputs =  inputs.transpose( 2, 3)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Security, skip this iteration if the batch_size is 1
                    if 1 == inputs.shape[0]:
                        file_log.write("Skipping iteration because batch_size = 1")
                        #print("Skipping iteration because batch_size = 1")
                        continue
                    # Debug
                    # debug_export_before_forward(inputs, labels, counter)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    iou_mean = iou2D(preds, labels, self.parameters["numero_classi"]).mean()
                    running_loss += loss.item() * inputs.size(0)
                    running_iou_means.append(iou_mean)
                    # Increment counter
                    counter = counter + 1

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                if running_iou_means is not None:
                    epoch_acc = np.array(running_iou_means).mean()
                else:
                    epoch_acc = 0.

                file_log.write('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'train':
                    loss_h.append(epoch_loss)
                    acc_h.append(epoch_acc)
                    if epoch>1:
                        ax[0].plot([epoch-1,epoch],loss_h[-2:], 'b')
                        ax[1].plot([epoch-1,epoch],acc_h[-2:], 'b')
                        ax[0].set_xlabel('Epochs', fontsize=14)
                        ax[0].set_ylabel('Loss', fontsize=14)
                        ax[1].set_xlabel('Epochs', fontsize=14)
                        ax[1].set_ylabel('Accuracy', fontsize=14)
                        #plt.pause(0.1)
                        plt.draw()

                if phase == 'val':
                    val_loss_h.append(epoch_loss)
                    val_acc_h.append(epoch_acc)
                    if epoch>1:
                        ax[0].plot([epoch-1,epoch],val_loss_h[-2:], 'r')
                        ax[1].plot([epoch-1,epoch],val_acc_h[-2:], 'r')
                        ax[0].set_xlabel('Epochs', fontsize=14)
                        ax[0].set_ylabel('Loss', fontsize=14)
                        ax[1].set_xlabel('Epochs', fontsize=14)
                        ax[1].set_ylabel('Accuracy', fontsize=14)
                        ax[0].legend(['Train_loss','Validation_loss'], fontsize=14)
                        ax[1].legend(['Train_accuracy','Validation_accuracy'], fontsize=14)
                        plt.savefig(self.output_dir + "epoch.png")
                        #plt.pause(0.1)
                        plt.draw()

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_state_dict = copy.deepcopy(model.state_dict())

                    #current_model_path = os.path.join(self.output_dir, f"checkpoint_bellissimo_DeepLabV3_Skydiver.pth")
                    #print(f"Save current model : {current_model_path}")
                    #torch.save(model.state_dict(), current_model_path)
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

                # Save current model every 25 epochs
                #if 0 == epoch%25:
                    #current_model_path = os.path.join(self.output_dir, f"checkpoint_{epoch:04}_DeepLabV3_Skydiver.pth")
                    #print(f"Save current model : {current_model_path}")
                    #torch.save(model.state_dict(), current_model_path)
                
            file_log.write("\n")
            file_log.close()
            #print()

        time_elapsed = time.time() - since
        file_log = open(self.dataset.dataset_info["directory_output"] + 'log_file.txt', 'a')
        file_log.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        file_log.write('Best val Acc: {:4f}'.format(best_acc))
        #print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        file_log.close()

        # load best model weights
        return best_model_state_dict, val_acc_history
    
    def inferenza(self, stack_topredict, axis):

        start_time = time.perf_counter()

        if self.parameters["data_type"] == "RGB":
            n_channels = 3
        else:
            n_channels = 1
        
        if self.parameters["network"] == "UNET":
            model = UNet(n_channels = n_channels, n_class = self.parameters["numero_classi"])

        elif "SEGNET" in self.parameters["network"]:
            if "VGG" in self.parameters["network"]:
                model = SegNet(input_channels = n_channels, output_channels = self.parameters["numero_classi"], VGG = True) #segnet
            elif "Continual" in self.parameters["network"]:
                model = Continual_SegNet(input_channels = n_channels, output_channels = self.parameters["numero_classi"])
            else:
                model = SegNet(input_channels = n_channels, output_channels = self.parameters["numero_classi"], VGG = False) #segnet

        weight_dir = self.dataset.dataset_info["directory_output"] + "pesi_modello.pth"
        state_dict = torch.load(weight_dir)
        model.load_state_dict(state_dict)

        if self.device == torch.device("cuda"):
            model = nn.DataParallel(model, device_ids=[0,1])
        
        model = model.to(self.device)
        model.eval()
        rig = stack_topredict.shape[0]
        col = stack_topredict.shape[1]
        zeta = stack_topredict.shape[2]
        stack_inference = np.empty([rig, col, zeta], dtype=np.uint8)
        transforms_image = transforms.ToTensor()

        if axis == "XY":
            for x in range(zeta):
                imm_estratta = stack_topredict[:, :, x]
                image = unfold_image(imm_estratta, self.parameters["tiles"], patches = False)
                image = image.to(self.device)
                outputs = model(image)
                _, preds = torch.max(outputs, 1)
                preds = preds.to("cpu")
                #preds = fold_image(preds, rig, col, self.parameters["tiles"])
                labels = preds.squeeze(0).cpu().numpy().astype(np.uint8)
                stack_inference[:, :, x] = np.copy(labels)

        elif axis == "XZ":
            for x in range(rig):
                imm_estratta = stack_topredict[x, :, :]
                image = unfold_image(imm_estratta, self.parameters["tiles"], patches = False)
                image = image.to(self.device)
                outputs = model(image)
                _, preds = torch.max(outputs, 1)
                preds = preds.to("cpu")
                #preds = fold_image(preds, col, zeta, self.parameters["tiles"])
                labels = preds.squeeze(0).cpu().numpy().astype(np.uint8)
                stack_inference[x, :, :] = np.copy(labels)

        elif axis == "YZ":
            for x in range(col):
                imm_estratta = stack_topredict[:, x, :]
                image = unfold_image(imm_estratta, self.parameters["tiles"], patches = False)
                image = image.to(self.device)
                outputs = model(image)
                _, preds = torch.max(outputs, 1)
                preds = preds.to("cpu")
                #preds = fold_image(preds, rig, zeta, self.parameters["tiles"])
                labels = preds.squeeze(0).cpu().numpy().astype(np.uint8)
                stack_inference[:, x, :] = np.copy(labels)

        del model
        finish_time = time.perf_counter()
        print("Immagini predette in {} seconds".format(finish_time - start_time))
        print("---")
        torch.cuda.reset_peak_memory_stats(device=0)
        torch.cuda.reset_max_memory_allocated(device=1)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return stack_inference

    def prediction(self, mode, subprocess = False):
            if subprocess == True:
                set_start_method('spawn', force=True)
                for grays in self.dataset.images:
                    if mode == "2D":
                        func_par = partial(self.inferenza, grays.np_stack)
                        with Pool(processes=1) as pool:
                            result = pool.map(func_par, ["XY"])
                        predizione = CT_stack(np.copy(result[0]), grays.voxel_size, "prediction", grays.name)
                        self.dataset.prediction.append(predizione)
                        self.dataset.dataset_info["prediction"].append(grays.name + "_prediction")

                    elif mode == "3D":
                        assi = ["XY", "XZ", "YZ"]
                        predizionis = []
                        func_par = partial(self.inferenza, grays.np_stack)
                        for asse in assi:
                            print(asse)
                            with Pool(processes=1) as pool:
                                result = pool.map(func_par, [asse])
                            predizionis.append(np.copy(result[0]))
                        pred_tot = calcola_moda_parallelized(predizionis)
                        self.dataset.prediction.append(CT_stack(np.copy(pred_tot), grays.voxel_size, "prediction", grays.name))
                        self.dataset.dataset_info["prediction"].append(grays.name + "_prediction")
            else:
                for grays in self.dataset.images:
                    if mode == "2D":
                        pred_result = self.inferenza(stack_topredict = grays.np_stack, axis = "XY" )
                        predizione = CT_stack(np.copy(pred_result), grays.voxel_size, "prediction", grays.name)
                        self.dataset.prediction.append(predizione)
                        self.dataset.dataset_info["prediction"].append(grays.name + "_prediction")

                    elif mode == "3D":
                        assi = ["XY", "XZ", "YZ"]
                        predizionis = []
                        for asse in assi:
                            print(asse)
                            pred_result = self.inferenza(stack_topredict = grays.np_stack, axis = asse)
                            predizionis.append(np.copy(pred_result))
                        pred_tot = calcola_moda_parallelized(predizionis)
                        self.dataset.prediction.append(CT_stack(np.copy(pred_tot), grays.voxel_size, "prediction", grays.name))
                        self.dataset.dataset_info["prediction"].append(grays.name + "_prediction")


    def grad_cam(self, stack_topredict):

        start_time = time.perf_counter()

        if self.parameters["data_type"] == "RGB":
            n_channels = 3
        else:
            n_channels = 1
        
        if self.parameters["network"] == "UNET":
            model = UNet(n_channels = n_channels, n_class = self.parameters["numero_classi"])

        elif "SEGNET" in self.parameters["network"]:
            if "VGG" in self.parameters["network"]:
                model = SegNet(input_channels = n_channels, output_channels = self.parameters["numero_classi"], VGG = True) #segnet
            elif "Continual" in self.parameters["network"]:
                model = Continual_SegNet(input_channels = n_channels, output_channels = self.parameters["numero_classi"])
            else:
                model = SegNet(input_channels = n_channels, output_channels = self.parameters["numero_classi"], VGG = False) #segnet

        weight_dir = self.dataset.dataset_info["directory_output"] + "pesi_modello.pth"
        state_dict = torch.load(weight_dir)
        model.load_state_dict(state_dict)

        if self.device == torch.device("cuda"):
            model = nn.DataParallel(model, device_ids=[0,1])
        
        model = model.to(self.device)
        model.eval()

        rig = stack_topredict.shape[0]
        col = stack_topredict.shape[1]
        zeta = stack_topredict.shape[2]
        stack_explain = np.empty([rig, col, zeta], dtype = stack_topredict.dtype)
        #img_to_tensor = transforms.ToTensor()
        feature_all = np.ones([rig, col], dtype = np.float32)
        feature_all_tensor = torch.tensor(feature_all, dtype=torch.float32)
        feature_all_tensor = feature_all_tensor.to(self.device)
        print("feature all ", feature_all_tensor.size())
        target_layers = [model.encoder_conv_42]
        targets = [SemanticSegmentationTarget(category = 1, mask = feature_all_tensor)]

        for x in range(zeta):
            imm_estratta = stack_topredict[:, :, x]
            image_tensor = torch.tensor(imm_estratta, dtype=torch.float32)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.unsqueeze(0)
            #print("image tensor", image_tensor.size())
            image_tensor = image_tensor.to(self.device)
            with GradCAM(model=model, target_layers = target_layers, use_cuda = True) as cam:
                feaures_maps = cam(input_tensor = image_tensor, targets = targets)
            grayscale_cam = feaures_maps[0, :]
            stack_explain[:,:,x] = np.copy(grayscale_cam)

        features_weights = stack_explain + stack_topredict
        features_weights = features_weights/features_weights.max()

        return features_weights

    def explain(self):
        for grays in self.dataset.images:
            funzione = self.grad_cam
            with Pool() as pool:
                result = pool.map(funzione, [grays])

            predizione = CT_stack(np.copy(result[0]), grays.voxel_size, "explained", grays.name)
            self.dataset.grad_CAM.append(predizione)
            self.dataset.dataset_info["explained"] = "Si"

    def save_info(self):
        with open(self.output_dir + self.dataset.dataset_info["modalità"] + "_parameters" + ".json", "w") as file:
            json.dump(self.parameters, file)

    def print_info(self):
        pprint.pprint(self.parameters)
    



        
    