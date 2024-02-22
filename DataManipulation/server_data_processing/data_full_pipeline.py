import data_utils
import os
import rasterio

# Setting the master folder (Which has subfolders image and landslide_data)

master_folder = 'LS_machine_learning_1region'
images_in_master = master_folder + '/Image'
landslides_in_master = master_folder + '/Landslide_Data'

# Output paths
multiband_outputs_folder = 'Processed_Data/Multibands'
mosaics_outputs_folder = 'Processed_Data/Mosaics'
image_patch_outputs_folder = 'Processed_Data/image_patches'
shapefile_patch_outputs = 'Processed_Data/shape_patches'
labels_yolo_path = 'Processed_Data/labels'

# Map from region to desired shapefile
region_to_shapefile={'1987 Sichuan pre-earthquake':landslides_in_master + '/1987 Sichuan rainstorm/preeqbefore1987.shp'}

##### MAIN PROCESSING #####

# Iterating through all the regions ( Eg. Sichuan, Kashmir, etc)
regions = [os.path.splitext(filename)[0] for filename in os.listdir(images_in_master) if not filename.startswith('.')]
regions_path = [images_in_master + '/' +  os.path.splitext(filename)[0] for filename in os.listdir(images_in_master) if not filename.startswith('.')]

#For Each region, we perform the following steps:
for region in regions:

    # Create folders with region names in Multibands folder
    # Check if the directory exists
    if not os.path.exists(multiband_outputs_folder + '/' + region):
        # If it doesn't exist, create it
        os.makedirs(multiband_outputs_folder + '/' + region)
        
    
    # Create folders with region names in Mosaics folder
    # Check if the directory exists
    if not os.path.exists(mosaics_outputs_folder + '/' + region):
        # If it doesn't exist, create it
        os.makedirs(mosaics_outputs_folder + '/' + region)

    areas_in_region = [os.path.splitext(filename)[0] for filename in os.listdir(images_in_master + '/' + region) if not filename.startswith('.')]
    
    # In each region, first form a multi band tif for each area.
    for area in areas_in_region:
        #print(area)
        area_folder_path = images_in_master + '/' + region + '/' + area
        #print(area_folder_path)

        # TIF band files in an area folder:
        band_names = [bandname for bandname in os.listdir(area_folder_path) if bandname.startswith(area) and bandname.lower().endswith('.tif')]
        #print(band_names)

        output_file_path = multiband_outputs_folder + '/' + region + '/' + area + '_Multiband.TIF'
        #print(output_file_path)

        data_utils.combine_bands_to_single_tif(bands_folder = area_folder_path, band_files = band_names, output_file = output_file_path)
        
    #print('multiband area creation done')
    ####### AFTER ALL AREA MULTIBANDS ARE GENERATED:
    # Now combine to form one big mosaic (join multiple area multiband tifs) for a region 

    # Get list of raster multibands of multiple areas belonging to this region
    multiband_region_folder = multiband_outputs_folder + '/' + region 
    region_multiband_list_paths = [multiband_region_folder + '/' + filename for filename in os.listdir(multiband_region_folder) if filename.lower().endswith('.tif')]
    
    # Path to save mosaic
    output_mosaic_region = mosaics_outputs_folder + '/' + region + '/' + region + '_mosaic.TIF'

    # Create the big mosaic
    data_utils.generate_mosiac_from_raster_list(raster_list = region_multiband_list_paths[1:3], output_path = output_mosaic_region)
    
    # We first obtain the right shapefile to use for the region
    desired_shapefile = region_to_shapefile[region]

    # We generate patches for the region mosaic file and shapefile
    data_utils.generate_patches_from_shapefile_and_raster_patches(input_tif = output_mosaic_region, input_shapefile = desired_shapefile, output_directory_images = image_patch_outputs_folder,output_directory_shapepatches = shapefile_patch_outputs, patch_size=(640, 640))
    
    ############# Preparing patches from mosaic and shapefile
    
    # We first obtain the right shapefile to use for the region
    desired_shapefile = region_to_shapefile[region]

    # We generate patches for the region mosaic file and shapefile
    data_utils.generate_patches_from_shapefile_and_raster_patches(input_tif = output_mosaic_region, input_shapefile = desired_shapefile, output_directory_images = image_patch_outputs_folder,output_directory_shapepatches = shapefile_patch_outputs,region_name = region, patch_size=(640, 640))
    
    ############# Preparing labels from shapefiles
    # We iterate through all shapefiles to generate yolo labels as .txt files
    # Obtain list of shapefiles in the shapefile_patch_outputs folder corresponding to this region
    shapefile_patch_list = [bandname for bandname in os.listdir(shapefile_patch_outputs) if bandname.startswith(region) and bandname.lower().endswith('.shp')]
    
    for shapefile_patch in shapefile_patch_list:
        shapefile_patch_path =  shapefile_patch_outputs + '/' + shapefile_patch
        labelfile_desired_path = labels_yolo_path + '/' + shapefile_patch[:-4] + '.txt'
        data_utils.generate_labels(shapefile_path = shapefile_patch_path,  target_label_path = labelfile_desired_path)

