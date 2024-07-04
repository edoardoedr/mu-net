import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import os
import glob
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


class DataLoaderSegmentation_gray(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, mode, tiles = 224):
        super(DataLoaderSegmentation_gray, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'img','*.*'))
        self.label_files = []
        self.tiles = tiles
        for img_path in self.img_files:
            image_filename, _ = os.path.splitext(os.path.basename(img_path))
            lab_filename = image_filename
            label_filename_with_ext = "label_"+lab_filename+".tif"
            self.label_files.append(os.path.join(folder_path, 'label', label_filename_with_ext))

        if "val" == mode :
            self.transforms = transforms.Compose([
                transforms.CenterCrop((self.tiles, self.tiles)),
                #transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms.Compose([
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(),
                    transforms.RandomCrop((self.tiles, self.tiles)),
                    #transforms.ToTensor(),
                ])
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
            img_path = self.img_files[index]
            label_path = self.label_files[index]
            img = Image.open(img_path)
            label = Image.open(label_path)

            if np.asarray(img).dtype == np.uint16:
                img_np = np.copy(img)
                img = img_np.astype(np.float32)/65535

            image_tensor = self.to_tensor(img)
            label_tensor = self.to_tensor(label)
            image_and_label_tensor = torch.stack([image_tensor, label_tensor])
            
            # Apply Transforms
            image_and_label_tensor = self.transforms(image_and_label_tensor)
           
            # Extract image and label
            image = image_and_label_tensor[0, 0, :, :]
            label = image_and_label_tensor[1, 0, :, :]
            image = image.unsqueeze(0)
            label = label * 255

            #  Convert to int64 and remove second dimension
            label = label.long()

            return image, label

    def __len__(self):
        return len(self.img_files)
    
class DataLoaderSAM(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, mode, tiles = 226, use_box = False, target_length = 226):
        super(DataLoaderSegmentation_gray, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'img','*.*'))
        self.label_files = []
        self.tiles = tiles
        self.use_box = use_box
        self.target_length = target_length
        
        for img_path in self.img_files:
            image_filename, _ = os.path.splitext(os.path.basename(img_path))
            lab_filename = image_filename
            label_filename_with_ext = "label_"+lab_filename+".tif"
            self.label_files.append(os.path.join(folder_path, 'label', label_filename_with_ext))

        if "val" == mode :
            self.transforms = transforms.Compose([
                transforms.CenterCrop((self.tiles, self.tiles)),
                #transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms.Compose([
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(),
                    transforms.RandomCrop((self.tiles, self.tiles)),
                    #transforms.ToTensor(),
                ])
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
            img_path = self.img_files[index]
            label_path = self.label_files[index]
            img = Image.open(img_path)
            label = Image.open(label_path)

            if np.asarray(img).dtype == np.uint16:
                img_np = np.copy(img)
                img = img_np.astype(np.float32)/65535

            image_tensor = self.to_tensor(img)
            label_tensor = self.to_tensor(label)
            image_and_label_tensor = torch.stack([image_tensor, label_tensor])
            
            # Apply Transforms
            image_and_label_tensor = self.transforms(image_and_label_tensor)
           
            # Extract image and label
            image = image_and_label_tensor[0, 0, :, :]
            label = image_and_label_tensor[1, 0, :, :]
            image = image.unsqueeze(0)
            label = label * 255

            #  Convert to int64 and remove second dimension
            label = label.long()
            
            image, label, boxes = self._process_image(image, label)

            return image, label, boxes

    def __len__(self):
        return len(self.img_files)
    
    def _process_image(self, image, gt2D):
            
        C, H, W = image.size()
            
        if C == 1:
            image = image.repeat(3, 1, 1)
                
        image_resized = self._resize_longest_side_image(image)
        mask_resized = self._resize_longest_side_mask(gt2D)
        img_padded = self._pad_image(image_resized)
        mask_padded = self._pad_mask(mask_resized)
                
        if not self.use_boxes:
            bboxes = np.array([0, 0, self.target_length, self.target_length])
            boxes = torch.tensor([bboxes], dtype=torch.int64).unsqueeze(1)
        else:
            boxes = self._get_bounding_boxes(mask_padded)
                
        return img_padded, mask_padded, boxes
        
    def _get_bounding_boxes(gt2D):
        # Inizializza un tensore per memorizzare le coordinate delle bounding box per ogni batch
        bounding_boxes = torch.zeros((1, 4), dtype=torch.int64)

        y_indices, x_indices = (gt2D > 0).nonzero(as_tuple=True)

        # Ottieni gli indici minimi e massimi per x e y
        x_min, x_max = torch.min(x_indices), torch.max(x_indices)
        y_min, y_max = torch.min(y_indices), torch.max(y_indices)

        # Assegna le coordinate al tensore delle bounding box
        bounding_boxes[0, :] = torch.tensor([x_min, y_min, x_max, y_max])

        # Restituisci il tensore delle bounding box
        return bounding_boxes
    
    def _resize_longest_side_image(self, images):
        """
        Expects a PyTorch tensor with shape [B, C, H, W].
        """
        C, oldh, oldw = images.shape
        scale = self.target_length * 1.0 / max(oldh, oldw)
        newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
        resize = transforms.Resize((newh, neww))
        resized_images = resize(images)
        return resized_images
    
    def _resize_longest_side_mask(self, masks):
        """
        Expects a PyTorch tensor with shape [B, C, H, W].
        """
        oldh, oldw = masks.shape
        scale = self.target_length * 1.0 / max(oldh, oldw)
        newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
        resize = transforms.Resize((newh, neww), interpolation = InterpolationMode.NEAREST)
        resized_masks = resize(masks)
        return resized_masks
    
    def _pad_image(self, images):
        """
        Expects a PyTorch tensor with shape [B, C, H, W].
        """
        C, h, w = images.shape
        padh = self.target_length - h
        padw = self.target_length - w
        # Pad the images
        padding = (0, padw, 0, padh)  # left, right, top, bottom
        padded_images = F.pad(images, padding, "constant", 0)
        return padded_images
    
    def _pad_mask(self, mask):
        """
        Expects a PyTorch tensor with shape [B, C, H, W].
        """
        h, w = mask.shape
        padh = self.target_length - h
        padw = self.target_length - w
        # Pad the images
        padding = (0, padw, 0, padh)  # left, right, top, bottom
        padded_masks = F.pad(mask, padding, "constant", 0)
        return padded_masks