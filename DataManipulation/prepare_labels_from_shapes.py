import os
import geopandas as gpd
import rasterio
import os
import numpy as np
import cv2

from data_utils import add_labels_to_folder
                                            
# Path to folder with shapefile_patches, tif_patches, labels
parent_folder = "Mosaic"

# Run the function
add_labels_to_folder(parent_folder)