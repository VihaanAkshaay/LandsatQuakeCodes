import numpy as np
from PIL import Image, ImageDraw
import h5py
import os
import torch
from torch.utils import data
from torch.utils.data import Dataset

class LandslideDataSet(data.Dataset):
    def __init__(self, data_dir, list_path, max_iters=None,set='label'):
        self.list_path = list_path
        self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        self.std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []

        if set=='labeled':
            for name in self.img_ids:
                img_file = data_dir + name
                label_file = data_dir + name.replace('img','mask').replace('image','mask')
                self.files.append({
                    'img': img_file,
                    'label': label_file,
                    'name': name
                })
        elif set=='unlabeled':
            for name in self.img_ids:
                img_file = data_dir + name
                self.files.append({
                    'img': img_file,
                    'name': name
                })
            
    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        
        if self.set=='labeled':
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            with h5py.File(datafiles['label'], 'r') as hf:
                label = hf['mask'][:]
            name = datafiles['name']
                
            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.float32)
            image = image.transpose((-1, 0, 1))
            size = image.shape

            for i in range(len(self.mean)):
                image[i,:,:] -= self.mean[i]
                image[i,:,:] /= self.std[i]

            return image.copy(), label.copy(), np.array(size), name

        else:
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            name = datafiles['name']
                
            image = np.asarray(image, np.float32)
            image = image.transpose((-1, 0, 1))
            size = image.shape

            for i in range(len(self.mean)):
                image[i,:,:] -= self.mean[i]
                image[i,:,:] /= self.std[i]

            return image.copy(), np.array(size), name

class CustomDataset_L4S_Split_Train_Test_Val(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.h5')]
        self.mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.h5')]
        assert len(self.img_files) == len(self.mask_files), "Number of images and masks do not match!"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        with h5py.File(self.img_files[idx], 'r') as file:
            img = torch.tensor(file['img'][:])
        with h5py.File(self.mask_files[idx], 'r') as file:
            mask = torch.tensor(file['mask'][:])

        img = img.permute(2, 0, 1)
        img = img.float()
        mask = mask.unsqueeze(0)
        mask = mask.float()
    
        return img, mask

class CustomDataset_L4S(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.h5')]
        self.mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.h5')]
        
        # This ensures that the file lists are in the same order
        self.img_files.sort()
        self.mask_files.sort()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        with h5py.File(self.img_files[idx], 'r') as file:
            img = torch.tensor(file['data'][:])
        
        with h5py.File(self.mask_files[idx], 'r') as file:
            mask = torch.tensor(file['data'][:])

        return img, mask
    
    

##### Defining a mapping function #####

def mapGeoToPixel(raster_image,coords):
    '''
    The input is the raster image & coordinates are in latitude and longitute. 
    '''
    bounds = raster_image.bounds
    x_min = bounds[0]
    x_max = bounds[2]
    y_min = bounds[1]
    y_max = bounds[3]
    
    new_coords = []
    
    #Converting coords (origially in lat-long space to x-y space in picture)
    
    for coord in coords:
        #print(coord)
        x_coord = int((coord[0]-x_min)*(raster_image.width)/(x_max - x_min))
        y_coord = int((coord[1]-y_min)*(raster_image.height)/(y_max - y_min))   
        new_coords.append((x_coord,y_coord))
    
    return new_coords

##### Shapefile to binary_mask #####
def shapefileToBinarymask(raster_image, shapefile):

    # Define the dimensions of the segmentation map
    width = raster_image.width
    height = raster_image.height

    # Create an empty binary segmentation map
    seg_map = np.zeros((height, width), dtype=np.uint8)

    # Create an Image object to draw on
    seg_map_image = Image.fromarray(seg_map)
    draw = ImageDraw.Draw(seg_map_image)

    # Define the coordinates of the polygon vertices
    for i in range(len(shapefile.geometry)):
        #polygon_list_latlong = [(x, y) for x, y,_ in shapefile.geometry[i].exterior.coords]
        polygon_list_latlong = [(x, y) for x, y,_ in shapefile.geometry[i].exterior.coords]
        #print(i)
        polygon_list = mapGeoToPixel(raster_image,polygon_list_latlong)

        # Draw the polygon on the segmentation map
        draw.polygon(polygon_list, fill=1)
    
    #convert back to np.array
    seg_map = np.array(seg_map_image)

    return seg_map

#All changes are done to 'draw' and so, to 'seg_map_image'

##### Obtain raster image from raster file #####
def obtainRasterImageSingleBand(raster_image):
    
    band_id = 1
    band_raster = raster_image.read(band_id)
    band_raster_flipped = np.flip(band_raster, axis=0)

    return band_raster_flipped
