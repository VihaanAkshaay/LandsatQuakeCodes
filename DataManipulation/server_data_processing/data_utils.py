
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio import merge
from rasterio.errors import RasterioIOError
from shapely.geometry import box
import cv2
import os

import matplotlib.pyplot as plt

# Display an image given the path
def display_image_jupyter_notebook(image_path):
    '''

    function to display an image using matplotlib in jupyter notebook
    image_path: path to image
    '''
    # display the image using matplotlib
    plt.imshow(plt.imread(image_path))
    plt.axis('off')
    plt.show()

def get_image_dimensions(image_path):
    '''
    function to get the dimensions of an image
    image_path: path to image
    '''
    # Open the image
    plt.imread(image_path)

    # Get the dimensions
    dimensions = plt.imread(image_path).shape

    return dimensions

def get_instances_number_yolo_format(label_path):
    '''
    function to get the number of instances in a label file in YOLO format
    label_path: path to label file
    '''
    # Open the label file
    with open(label_path,'r') as f:
        labels = [line.rstrip('\n') for line in open(label_path)]

    return len(labels)

def get_bounding_box_from_yolo_label(flat_coordinates, box_format='xyxy'):
    '''
    function to get the bounding box from a label file in YOLO format
    label: label in YOLO (instance segmentation) format - list of floats between 0 and 1

    return: bounding box (YOLO SCALE)
    '''

    # Separate x and y values:
    x_min = flat_coordinates[0]
    y_min = flat_coordinates[1]
    x_max = flat_coordinates[0]
    y_max = flat_coordinates[1]

    for i in range(len(flat_coordinates)):

        # x values
        if i % 2 == 0:
            if flat_coordinates[i] < x_min:
                x_min = flat_coordinates[i]
            elif flat_coordinates[i] > x_max:
                x_max = flat_coordinates[i]

        # y values
        else:
            if flat_coordinates[i] < y_min:
                y_min = flat_coordinates[i]
            elif flat_coordinates[i] > y_max:
                y_max = flat_coordinates[i]

    if box_format == 'xyxy':
        return [x_min, y_min, x_max, y_max]
    
    elif box_format == 'xywh':
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    
    else:
        raise ValueError('Invalid box format. Use xyxy or xywh')

# Function to convert lat long coordinates of shapefile to normalised pixel coordinates of image
def convert_coordinates_to_pixel_values(x,y, shapefile_left, shapefile_bottom, shapefile_right, shapefile_top):
    '''
    x: longitude
    y: latitude
    shapefile_left: left coordinate of shapefile
    shapefile_bottom: bottom coordinate of shapefile
    shapefile_right: right coordinate of shapefile
    shapefile_top: top coordinate of shapefile
    
    Returns: x_pixel, y_pixel
    '''

    x_pixel = (x - shapefile_left) / (shapefile_right - shapefile_left)
    y_pixel = (shapefile_top - y) / (shapefile_top - shapefile_bottom)
    return x_pixel, y_pixel

def convert_list_coordinates_to_pixel_normalised(list,shapefile_left, shapefile_bottom, shapefile_right, shapefile_top):
    flat_list = []
    for val in list:
        x = val[0]
        y = val[1]
        x_pixel,y_pixel = convert_coordinates_to_pixel_values(x,y,shapefile_left, shapefile_bottom, shapefile_right, shapefile_top)
        flat_list.append(x_pixel)
        flat_list.append(y_pixel)

    return flat_list

def add_labels_to_folder(foldername):
    '''
    The folder should contain the following:
    1. shapefile_patches folder
    2. tif_patches folder
    3. labels folder'''

    # Checking all files in the image_patches folder:
    filenames = [os.path.splitext(filename)[0] for filename in os.listdir(foldername + "/image_patches")]

    total_len = len(filenames)
    counter = 0

    for picname in filenames:
        generate_labels(picname,foldername)
        counter += 1

    #print(counter, '/', total_len, 'done')  

def generate_labels(shapefile_path,  target_label_path):
    '''
    Function to generate labels (YOLO style) for a given shapefile name
    '''

    shapefile = gpd.read_file(shapefile_path)

    # If shapefile is empty, create an empty txt file and return
    if shapefile.empty:
        with open(target_label_path,'w') as f:
            f.write("")
        return

    # Getting bounds of the shapefile
    shapefile_left = shapefile['left'][0]
    shapefile_bottom = shapefile['bottom'][0]
    shapefile_right = shapefile['right'][0]
    shapefile_top = shapefile['top'][0]

    master_list = []

    # Define the coordinates of the polygon vertices
    for i in range(len(shapefile.geometry)):
        # Add attribute error try catch
        try:
            polygon_list_latlong = [(x, y) for x, y in shapefile.geometry[i].exterior.coords]
            #print(polygon_list_latlong)
            #print(convert_list_coordinates_to_pixel_normalised(polygon_list_latlong,raster_patch_path))
            master_list.append(convert_list_coordinates_to_pixel_normalised(polygon_list_latlong,shapefile_left, shapefile_bottom, shapefile_right, shapefile_top))
        except AttributeError:
            continue
        #print(i)
        #polygon_list = mapGeoToPixel(raster_image,polygon_list_latlong)

    # Write the coordinates to a txt file with the same name as the shapefile and each entry in the lists are separated by a space and each list starts in a new line
    with open(target_label_path,'a') as f:
        for item in master_list:
            f.write("0")
            for val in item:
                # Save only upto 6 decimal points
                f.write(" " + str(round(val,6)))
            f.write("\n")

# Convert string seperated by space to a list of floats
def convert_string_list_to_float(lst):
    return list(map(float, lst.split()))

# Visualising an image and labels
def print_yolo_labels_on_image(image_path, yolo_label_format):
    '''
    image_path: path to image
    yolo_label_format: path to label file
    '''

    # Load image
    image = cv2.imread(image_path)

    # Load labels
    with open(yolo_label_format,'r'):
        labels = [line.rstrip('\n') for line in open(yolo_label_format)]

    # Get image dimensions
    image_height, image_width, _ = image.shape

    # Scaling the labels (x labels are scaled by image_width and y labels are scaled by image_height)
    x_scale = image_width
    y_scale = image_height

    # Loop through labels
    for label in labels:

        # Converting string labels list to integer
        flat_coordinates = convert_string_list_to_float(label)[1:]

        # Scaling up the coordinates to x_scale and y_scale
        for i in range(len(flat_coordinates)):

            # x values
            if i % 2 == 0:
                flat_coordinates[i] *= x_scale
            # y values
            else:
                flat_coordinates[i] *= y_scale

        # Draw polygon
        polygon_coordinates = np.array(flat_coordinates).reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(image, [polygon_coordinates], True, (0, 255, 0), 1)

    # Display image
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Visualising an image and labels
def print_yolo_labels_on_image_jupyter_notebook(image_path, yolo_label_format):
    '''
    function to display an image using matplotlib in jupyter notebook
    image_path: path to image
    yolo_label_format: path to label file
    '''

    # Load image
    image = cv2.imread(image_path)

    # Load labels
    with open(yolo_label_format,'r'):
        labels = [line.rstrip('\n') for line in open(yolo_label_format)]

    # Get image dimensions
    image_height, image_width, _ = image.shape

    # Scaling the labels (x labels are scaled by image_width and y labels are scaled by image_height)
    x_scale = image_width
    y_scale = image_height
    
    # Loop through labels
    for label in labels:
            
            # Converting string labels list to integer
            flat_coordinates = convert_string_list_to_float(label)[1:]
    
            # Scaling up the coordinates to x_scale and y_scale
            for i in range(len(flat_coordinates)):
    
                # x values
                if i % 2 == 0:
                    flat_coordinates[i] *= x_scale
                # y values
                else:
                    flat_coordinates[i] *= y_scale
    
            # Draw polygon
            polygon_coordinates = np.array(flat_coordinates).reshape((-1,1,2)).astype(np.int32)
            cv2.polylines(image, [polygon_coordinates], True, (0, 255, 0), 1)

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Visualising an image and labels
def print_yolo_labels_on_numpy_image_jupyter_notebook(numpy_image_path, yolo_label_format):
    '''
    function to display an image using matplotlib in jupyter notebook
    image_path: path to image
    yolo_label_format: path to label file
    '''

    # Read all bands of the image
    numpy_image_all_bands = np.load(numpy_image_path)
    numpy_image_3bands = numpy_image_all_bands[:3]
    numpy_image = np.moveaxis(numpy_image_3bands, 0, -1)

    # Normalising the image:
    if numpy_image.dtype == np.uint16:
        scale_factor = 65535 // 255
        numpy_image = (numpy_image / scale_factor).astype(np.uint8)
    elif numpy_image.max() <= 1:  # float images in [0, 1] range
        numpy_image = (numpy_image * 255).astype(np.uint8)

    # Convert the image from numpy to cv2 imread format
    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    # Load labels
    with open(yolo_label_format,'r'):
        labels = [line.rstrip('\n') for line in open(yolo_label_format)]

    # Get image dimensions
    image_height, image_width, _ = image.shape

    # Scaling the labels (x labels are scaled by image_width and y labels are scaled by image_height)
    x_scale = image_width
    y_scale = image_height
    
    # Loop through labels
    for label in labels:
            
            # Converting string labels list to integer
            flat_coordinates = convert_string_list_to_float(label)[1:]
    
            # Scaling up the coordinates to x_scale and y_scale
            for i in range(len(flat_coordinates)):
    
                # x values
                if i % 2 == 0:
                    flat_coordinates[i] *= x_scale
                # y values
                else:
                    flat_coordinates[i] *= y_scale
    
            # Draw polygon
            polygon_coordinates = np.array(flat_coordinates).reshape((-1,1,2)).astype(np.int32)
            cv2.polylines(image, [polygon_coordinates], True, (0, 255, 0), 1)

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def how_empty_is_the_patch(patch,is_path=False):
    '''
    function to check how empty a patch is
    patch: patch to check
    '''

    if is_path:
        patch = np.load(patch)

    # Count the number of zeros
    zeros = np.count_nonzero(patch == 0)

    # Count the number of non-zeros
    non_zeros = np.count_nonzero(patch)

    # Calculate the percentage of zeros
    percentage_of_zeros = (zeros / (zeros + non_zeros)) * 100

    return percentage_of_zeros

def save_raster_patch_as_image(patch, output_filename,file_format):

    # If there are more than 3 bands, keep only the first 3 bands
    if patch.shape[0] > 3:
        patch = patch[:3, :, :]

    # Convert data range to [0, 255] for JPG images.
    if patch.dtype == np.uint16:
        scale_factor = 65535 // 255
        patch = (patch / scale_factor).astype(np.uint8)
    elif patch.max() <= 1:  # float images in [0, 1] range
        patch = (patch * 255).astype(np.uint8)

    # Save the patch as a new JPG file.
    with rasterio.open(output_filename, 'w', driver=file_format ,height=patch.shape[1], width=patch.shape[2], count=patch.shape[0], dtype=patch.dtype) as dest:
        dest.write(patch)

def save_raster_patch_as_numpy(patch, output_filename):

    '''
    patch: patch to save as numpy
    '''

    np.save(output_filename, patch)
        

def generate_patches_from_shapefile_and_raster_patches(input_tif, input_shapefile, output_directory_images,output_directory_shapepatches, region_name,file_format, numpy_output ,patch_size=(640, 640)):

    # Read the shapefile using Geopandas.
    gdf = gpd.read_file(input_shapefile)
    
    with rasterio.open(input_tif) as src:
        width, height = src.width, src.height
        
        for x in range(0, width, patch_size[0]):
            for y in range(0, height, patch_size[1]):

                # If the patch would exceed the image boundaries, skip it.
                if x + patch_size[0] > width or y + patch_size[1] > height:
                    continue
                
                # Calculate patch bounds.
                transform = src.window_transform(Window(col_off=x, row_off=y, width=patch_size[0], height=patch_size[1]))
                left, bottom, right, top = rasterio.transform.array_bounds(height=patch_size[1], width=patch_size[0], transform=transform)
                
                # Read the raster patch.
                patch = src.read(window=Window(col_off=x, row_off=y, width=patch_size[0], height=patch_size[1]))
                
                # Create a bounding box from the patch bounds.
                bounding_geometry = box(left, bottom, right, top)
                
                # Crop the shapefile using the bounding box.
                cropped_gdf = gdf[gdf.geometry.intersects(bounding_geometry)].copy()

                # Conditions to drop the sample ->

                # If the cropped shapefile is empty, skip it.
                if cropped_gdf.empty:
                    continue
                
                # If patch is mostly empty, skip it.
                if how_empty_is_the_patch(patch) > 50:
                    continue

                # Add bounds as attributes
                cropped_gdf['left'] = left
                cropped_gdf['right'] = right
                cropped_gdf['top'] = top
                cropped_gdf['bottom'] = bottom
                
                for index, row in cropped_gdf.iterrows():
                    if row['geometry'].is_valid and bounding_geometry.is_valid:
                        try:
                            cropped_gdf.at[index, 'geometry'] = row['geometry'].intersection(bounding_geometry)
                        except:
                            print('Error in intersection')
                    else:
                        print('Invalid geometry in index:', index)
                # Save the cropped shapefile based on its position.
                shapefile_output_filename = f"{output_directory_shapepatches}/{region_name}_patch_{x}_{y}.shp"
                cropped_gdf.to_file(shapefile_output_filename)

                # Save the raster patch 
                if numpy_output:
                    raster_output_filename = f"{output_directory_images}/{region_name}_patch_{x}_{y}.npy"
                    save_raster_patch_as_numpy(patch, raster_output_filename)
                else:
                    raster_output_filename = f"{output_directory_images}/{region_name}_patch_{x}_{y}.{file_format}"
                    save_raster_patch_as_image(patch, raster_output_filename,file_format)


def combine_bands_to_single_tif(bands_folder, band_files, band_numbers, output_file):
    '''
    bands_folder: folder containing all the bands
    band_files: list of band files
    band_numbers: list of band numbers or False to include all bands
    output_file: output file name
    '''

    # Modify the band_files list to only include the bands in band_numbers - given that the last character of the band file name is the band number
    if band_numbers:
        band_files = [band for band in band_files if int(band[-5]) in band_numbers]

    # Open the first band.
    with rasterio.open(f"{bands_folder}/{band_files[0]}") as src:
        # Read metadata from the first band.
        meta = src.meta

        # Update metadata to reflect the number of layers.
        meta.update(count = len(band_files))

        # Read each layer and write it to stack.
        with rasterio.open(output_file, 'w', **meta) as dst:
            for id, layer in enumerate(band_files, start=1):
                try:
                    with rasterio.open(f"{bands_folder}/{layer}") as src1:
                        dst.write_band(id, src1.read(1))
                except RasterioIOError:
                    continue


        # Close the files.
        src.close()
        dst.close()

def combine_rasters_to_big_mosaic(raster_folder, raster_files, output_file):
    '''
    raster_folder: folder containing all the rasters
    raster_files: list of raster files
    output_file: output file name
    '''
    # Open the first raster.
    with rasterio.open(f"{raster_folder}/{raster_files[0]}") as src:
        # Read metadata from the first raster.
        meta = src.meta

        # Update metadata to reflect the number of layers.
        meta.update(count = len(raster_files))

        # Read each layer and write it to stack.
        with rasterio.open(output_file, 'w', **meta) as dst:
            for id, layer in enumerate(raster_files, start=1):
                with rasterio.open(f"{raster_folder}/{layer}") as src1:
                    dst.write_band(id, src1.read(1))

        # Close the files.
        src.close()
        dst.close()

# Merging areas to form a bigger area (mosaic)
#(https://medium.com/spatial-data-science/how-to-mosaic-merge-raster-data-in-python-fb18e44f3c8)
def generate_mosiac_from_raster_list(raster_list,output_path):

    mosaic, output = merge.merge(raster_list)
    
    # Adjust the meta file
    temp_raster = rasterio.open(raster_list[0])
    output_meta = temp_raster.meta.copy()
    output_meta.update({"driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": output,})
    
    # Save the mosaic raster file with meta data
    with rasterio.open(output_path,'w', **output_meta) as m:
        m.write(mosaic)
    