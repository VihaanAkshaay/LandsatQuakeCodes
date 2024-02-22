import data_utils
import os
import rasterio

# Setting the master folder (Which has subfolders image and landslide_data) 
# Use pwd from terminal to get the current working directory

master_folder = '/Users/vihaan/Workspace/!Datasets/LS_Machine_Learning'
images_in_master = master_folder + '/Image'
landslides_in_master = master_folder + '/Landslide_Data'
desired_output_folder = '/Users/vihaan/Workspace/!Datasets/Processed_Data_new'

# Start the output folder fresh.
if os.path.exists(desired_output_folder):
    print('clearing old output folder')
    os.system('rm -rf ' + desired_output_folder)
    os.mkdir(desired_output_folder)


# Output paths - if they don't exist, create them

if not os.path.exists(desired_output_folder):
    os.mkdir(desired_output_folder)

if not os.path.exists(desired_output_folder + '/Multibands'):
    os.mkdir(desired_output_folder + '/Multibands')

if not os.path.exists(desired_output_folder + '/Mosaics'):
    os.mkdir(desired_output_folder + '/Mosaics')

if not os.path.exists(desired_output_folder + '/image_patches'):
    os.mkdir(desired_output_folder + '/image_patches')

if not os.path.exists(desired_output_folder + '/shape_patches'):
    os.mkdir(desired_output_folder + '/shape_patches')

if not os.path.exists(desired_output_folder + '/labels'):
    os.mkdir(desired_output_folder + '/labels')


multiband_outputs_folder = desired_output_folder + '/Multibands'
mosaics_outputs_folder = desired_output_folder + '/Mosaics'
image_patch_outputs_folder = desired_output_folder + '/image_patches'
shapefile_patch_outputs = desired_output_folder + '/shape_patches'
labels_yolo_path = desired_output_folder + '/labels'

# Set Desired Parameters
# Only use bands 1,2,3,4,5 and 7 (omit band 6)
DESIRED_BAND_LIST = [1,2,3,4,5,7]
# Image size 224 x 224
DESIRED_PATCH_SIZE = (224, 224)
# Save output as numpy arrays:
DESIRED_NUMPY_OUTPUT = True

# Map from region to desired shapefile
# Use only Sichuan pre 1987, Chamoli, Chichi, Sikkim
region_to_shapefile={'1987 Sichuan pre-earthquake':landslides_in_master + '/1987 Sichuan rainstorm/preeqbefore1987.shp',
                    '1999 chamoli earthquake':landslides_in_master + '/1999 chamoli earthquake/chomali_area.shp',
                    '1999 chichi earthquake':landslides_in_master + '/1999 chichi earthquake/ChiChi_1999_area.shp',
                    '2011 sikkim earthquake':landslides_in_master + '/2011 sikkim earthquake/sikkim_area.shp'}

##### MAIN PROCESSING #####

# Iterating through all the regions ( Eg. Sichuan, Kashmir, etc)
regions = [os.path.splitext(filename)[0] for filename in os.listdir(images_in_master) if not filename.startswith('.')]
regions_path = [images_in_master + '/' +  os.path.splitext(filename)[0] for filename in os.listdir(images_in_master) if not filename.startswith('.')]


#For Each region, we perform the following steps:
for region in regions:
    

    if region not in region_to_shapefile.keys():
        continue

    print('Starting the process for region: ' + region + '...')

    # Create folders with region names in Multibands folder
    # Check if the directory exists
    if not os.path.exists(multiband_outputs_folder + '/' + region):
        # If it doesn't exist, create it
        os.makedirs(multiband_outputs_folder + '/' + region)

    areas_in_region = [os.path.splitext(filename)[0] for filename in os.listdir(images_in_master + '/' + region) if not filename.startswith('.')]
    
    # In each region, first form a multi band tif for each area.
    for area in areas_in_region:

        # If area is not a folder, skip
        if not os.path.isdir(images_in_master + '/' + region + '/' + area):
            continue

        # Only consider areas that start with LT05 ot LT04
        if not area.startswith('LT05') and not area.startswith('LT04'):
            continue

        #print(area)
        area_folder_path = images_in_master + '/' + region + '/' + area
        #print(area_folder_path)

        # TIF band files in an area folder:
        band_names = [bandname for bandname in os.listdir(area_folder_path) if bandname.startswith(area) and bandname[-6] == 'B' and bandname.lower().endswith('.tif')]
        #print(band_names)

        output_file_path = multiband_outputs_folder + '/' + region + '/' + area + '_Multiband.TIF'
        #print(output_file_path)

        print(area_folder_path)

        data_utils.combine_bands_to_single_tif(bands_folder = area_folder_path, band_files = band_names, band_numbers = DESIRED_BAND_LIST, output_file = output_file_path)
        
    print('multiband area creation done for' + region + '...')

    ####### AFTER ALL AREA MULTIBANDS ARE GENERATED:
    # Now combine to form one big mosaic (join multiple area multiband tifs) for a region 

    # Get list of raster multibands of multiple areas belonging to this region

    # If multiband region folder doesn't exist, create it
    if not os.path.exists(mosaics_outputs_folder + '/' + region):
        os.makedirs(mosaics_outputs_folder + '/' + region)

    multiband_region_folder = multiband_outputs_folder + '/' + region 
    region_multiband_list_paths = [multiband_region_folder + '/' + filename for filename in os.listdir(multiband_region_folder) if filename.lower().endswith('.tif')]
    
    # Path to save mosaic
    output_mosaic_region = mosaics_outputs_folder + '/' + region + '/' + region + '_mosaic.TIF'

    # Create the big mosaic
    data_utils.generate_mosiac_from_raster_list(raster_list = region_multiband_list_paths[1:3], output_path = output_mosaic_region)
    
    print('mosaic creation done for' + region + '...')

    # We first obtain the right shapefile to use for the region
    desired_shapefile = region_to_shapefile[region]

    # We generate patches for the region mosaic file and shapefile
    data_utils.generate_patches_from_shapefile_and_raster_patches(input_tif = output_mosaic_region, input_shapefile = desired_shapefile, output_directory_images = image_patch_outputs_folder,output_directory_shapepatches = shapefile_patch_outputs,region_name= region, file_format = '.npy', numpy_output = DESIRED_NUMPY_OUTPUT, patch_size=DESIRED_PATCH_SIZE)
    
    print('patches creation done for' + region + '...')

    ############# Preparing labels from shapefiles
    # We iterate through all shapefiles to generate yolo labels as .txt files
    # Obtain list of shapefiles in the shapefile_patch_outputs folder corresponding to this region
    shapefile_patch_list = [bandname for bandname in os.listdir(shapefile_patch_outputs) if bandname.startswith(region) and bandname.lower().endswith('.shp')]
    
    for shapefile_patch in shapefile_patch_list:
        shapefile_patch_path =  shapefile_patch_outputs + '/' + shapefile_patch
        labelfile_desired_path = labels_yolo_path + '/' + shapefile_patch[:-4] + '.txt'
        data_utils.generate_labels(shapefile_path = shapefile_patch_path,  target_label_path = labelfile_desired_path)

