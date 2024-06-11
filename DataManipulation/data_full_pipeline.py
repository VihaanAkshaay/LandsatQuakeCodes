import data_utils
import os
import rasterio

# Setting the master folder (Which has subfolders image and landslide_data) 
# Use pwd from terminal to get the current working directory

master_folder = '/mnt/taurus/data2/vihaan/LS_machine_learning_S24'
images_in_master = master_folder + '/Image'
landslides_in_master = master_folder + '/Landslide_data'
desired_output_folder = '/mnt/taurus/data2/vihaan/Processed_Data_S24_with_DEM'
dem_mosaic_folder = '/mnt/taurus/data2/vihaan/DEM_Mosaics'

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
region_to_shapefile={'1984 Nagano':landslides_in_master + '/Nagano 1984/Nagano_1984_area.shp',
                    '1987 Sichuan pre-earthquake':landslides_in_master + '/1987 Sichuan rainstorm/preeqbefore1987_area.shp',
                    '1991 Limon Earthquake': landslides_in_master + '/Valle de la Estrella 1991/ValleEstrella_2016_area.shp',
                    '1998 Jueili':landslides_in_master + '/Jueili 1998/Jueili_1999_area.shp',
                    '1999 chamoli earthquake':landslides_in_master + '/1999 chamoli earthquake/chomali_area.shp',
                    '1999 chichi earthquake':landslides_in_master + '/1999 chichi earthquake/ChiChi_1999_area.shp',
                    '2004 Cheutsu Earthquake 2004': landslides_in_master + '/Niigata 2004/Niigata_2006_area.shp',
                    '2005 Kashmir earthquake':landslides_in_master + '/2005 Kashmir earthquake/Kashmir_Basharat2016_area.shp',
                    '2008 Iwate':landslides_in_master + '/EHonshu 2008/EHonshu_2019_area.shp',
                    '2010 Haiti':landslides_in_master + '/Haiti 2010/Haiti_2013_area.shp',
                    '2011 Arun rainstorm':landslides_in_master + '/2011 Arun rainstorm/2011_arun_area.shp',
                    '2011 sikkim earthquake':landslides_in_master + '/2011 sikkim earthquake/sikkim_area.shp',
                    '2012 Arun rainstorm': landslides_in_master + '/2012 Arun rainstorm/2012_arun_area.shp',
                    '2013 Arun rainstorm':landslides_in_master + '/2013 Arun rainstorm/2013_arun_area.shp',
                    '2014 Arun rainstorm':landslides_in_master + '/2014 Arun rainstorm/2014_arun_area.shp',
                    '2014 Ludian':landslides_in_master + '/Sichuan 2014/Sichuan_2015a_area.shp',
                    '2015 Arun rainstorm':landslides_in_master + '/2015 Arun rainstorm/2015_arun_area.shp',
                    '2015 gorkha earthquake':landslides_in_master + '/2015 gorkha earthquake/full_edit_final_area.shp',
                    '2016 Arun rainstorm':landslides_in_master + '/2016 Arun rainstorm/2016_arun_area.shp',
                    '2017 Arun rainstorm':landslides_in_master + '/2017 Arun rainstorm/2017_arun_area.shp',
                    '2017 Milin':landslides_in_master + '/Milin 2017/Milin_2019_area.shp',
                    '2018 Arun rainstorm':landslides_in_master + '/2018 Arun rainstorm/2018_arun_area.shp',
                    '2018 Lombok':landslides_in_master + '/Lombok 2018/Lombok_2019a_area.shp',
                    'PapuaNG 2018':landslides_in_master + '/PapuaNG 2018/PapuaNG_2020_area.shp',
                    'Sulawesi 2018':landslides_in_master + '/Palu 2018/Palu_2021_area.shp',
                    '2019 Arun rainstorm':landslides_in_master + '/2019 Arun rainstorm/2019_arun_area.shp',
                    '2019 Cinchona':landslides_in_master + '/Cinchona 2019/Cinchona_2019_area.shp', 
                    'Mesatas 2019':landslides_in_master + '/Mesetas 2019/Mesetas_2019_area.shp',
                    '2020 Arun rainstorm':landslides_in_master + '/2020 Arun rainstorm/2020_arun_area.shp',
                    '2020 Capellades':landslides_in_master + '/Capellades 2020/Capellades_2020_area.shp',
                    }

region_to_demfile={'1984 Nagano':'1984 Nagano',
                    '1987 Sichuan pre-earthquake':'1987 sichuan pre-eq',
                    '1991 Limon Earthquake': '1991 Limon',
                    '1998 Jueili':'1998 Jueili',
                    '1999 chamoli earthquake':'1999 chamoli',
                    '1999 chichi earthquake':'1999 ChiChi',
                    '2004 Cheutsu Earthquake 2004': '2004 Chuetsu',
                    '2005 Kashmir earthquake':'2005 Kashmir',
                    '2008 Iwate':'2008 Iwate',
                    '2010 Haiti':'2010 Haiti',
                    '2011 Arun rainstorm':'2011_2020 arun',
                    '2011 sikkim earthquake':'2011 Sikkim',
                    '2012 Arun rainstorm':'2011_2020 arun',
                    '2013 Arun rainstorm':'2011_2020 arun',
                    '2014 Arun rainstorm':'2011_2020 arun',
                    '2014 Ludian':'2014 Ludian',
                    '2015 Arun rainstorm':'2011_2020 arun',
                    '2015 gorkha earthquake':'2015 gorkha',
                    '2016 Arun rainstorm':'2011_2020 arun',
                    '2017 Arun rainstorm':'2011_2020 arun',
                    '2017 Milin':'2017 Milin',
                    '2018 Arun rainstorm':'2011_2020 arun',
                    '2018 Lombok':'2018 Lombok',
                    'PapuaNG 2018':'2018 PapuaNG',
                    'Sulawesi 2018':'2018 Sulawesi',
                    '2019 Arun rainstorm':'2011_2020 arun',
                    '2019 Cinchona':'2009 Cinchona', 
                    'Mesatas 2019':'2019 Mesetas',
                    '2020 Arun rainstorm':'2011_2020 arun',
                    '2020 Capellades':'2016 Capellades',
                    }


# Check if all the folders and the files from the map above exist:
#for region in region_to_shapefile.keys():
#    if not os.path.exists(region_to_shapefile[region]):
#        print('Shapefile not found for region: ' + region)
#        print('Path: ' + region_to_shapefile[region])
#        exit(1)

##### MAIN PROCESSING #####

# Iterating through all the regions ( Eg. Sichuan, Kashmir, etc)
regions = [os.path.splitext(filename)[0] for filename in os.listdir(images_in_master) if not filename.startswith('.')]
regions_path = [images_in_master + '/' +  os.path.splitext(filename)[0] for filename in os.listdir(images_in_master) if not filename.startswith('.')]

# Print the total number of feasible regions
available_regions = [region for region in regions if region in region_to_shapefile.keys()]
region_count = 0

print("List of Regions: ", available_regions)

#For Each region, we perform the following steps:
for region in reversed(available_regions):

    # We ignore some regions:
    if region == '1999 chichi earthquake':
        print('skipping region',region)
        continue

    ### Check if DEM and images have same CRS, if not, print and move on to the next region:
    # Reading the crs of the dem:
    #current_DEM = dem_mosaic_folder + '/' + region_to_demfile[region] + '_mosaic.TIF'
    #temp_src = rasterio.open(current_DEM)
    #crs_dem = temp_src.crs

    # Region troubleshooting (ignore)
   
    # Progress:
    region_count += 1

    # Check if region is in the map
    if region not in region_to_shapefile.keys():
        print('Region not found in the shapefile map: ' + region)
        continue
    
    # Print region start with count and total
    print('Processing region: ' + region + ' (' + str(region_count) + '/' + str(len(available_regions)) + ')')

    # Create folders with region names in Multibands folder
    # Check if the directory exists
    if not os.path.exists(multiband_outputs_folder + '/' + region):
        # If it doesn't exist, create it
        os.makedirs(multiband_outputs_folder + '/' + region)
    
    # Check if region already has image patches, if yes, we skip to the next region
    if os.path.exists(image_patch_outputs_folder + '/' + region):
        print('Skipping region:' + region + 'as image patches already exists')
        continue

    areas_in_region = [os.path.splitext(filename)[0] for filename in os.listdir(images_in_master + '/' + region) if not filename.startswith('.')]


    # In each region, first form a multi band tif for each area.
    for area in areas_in_region:

        # If area is not a folder, skip
        if not os.path.isdir(images_in_master + '/' + region + '/' + area):
            continue

        # Only consider areas that start with LT05 or LT04 or LC08
        if not area.startswith('LT05') and not area.startswith('LT04') and not area.startswith('LC08'):
            continue

        #print(area)
        area_folder_path = images_in_master + '/' + region + '/' + area
        #print(area_folder_path)

        # TIF band files in an area folder:
        band_names = [bandname for bandname in os.listdir(area_folder_path) if bandname.startswith(area) and bandname[-6] == 'B' and bandname.lower().endswith('.tif')]
        
        # If there are no band files, skip
        if len(band_names) == 0:
            continue

        output_file_path = multiband_outputs_folder + '/' + region + '/' + area + '_Multiband.TIF'
        #print(output_file_path)

        #print(area_folder_path)

        data_utils.combine_bands_to_single_tif(bands_folder = area_folder_path, band_files = band_names, band_numbers = DESIRED_BAND_LIST, output_file = output_file_path)
        


    # If no multibands were created, skip to the next region
    if len([filename for filename in os.listdir(multiband_outputs_folder + '/' + region) if filename.lower().endswith('.tif')]) == 0:
        print('No multibands created for region: ' + region)
        continue

    print('multibands creation done for' + region + '...')

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
    #data_utils.generate_mosiac_from_raster_list(raster_list = region_multiband_list_paths[1:3], output_path = output_mosaic_region)
    data_utils.generate_mosiac_from_raster_list(raster_list = region_multiband_list_paths, output_path = output_mosaic_region)
    
    print('mosaic creation done for' + region + '...')

    # We first obtain the right shapefile to use for the region
    desired_shapefile = region_to_shapefile[region]

    # Obtain the corresponding DEM mosaic for this particular region:
    current_DEM = dem_mosaic_folder + '/' + region_to_demfile[region] + '_mosaic.TIF'

    ### Check if DEM and mosaic have the same crs:
    # Reading the mosaic's crs
   # mosaic_src = rasterio.open(output_mosaic_region)
   # mosaic_crs = mosaic_src.crs

    #if mosaic_crs != crs_dem:
    #    print('CRS mismatch for region:', region)
    #    print('-------------------------!!!!!!!--------------------------')
    #    continue

    # We generate patches for the region mosaic file and shapefile
    data_utils.generate_patches_from_shapefile_and_raster_patches(input_tif = output_mosaic_region, input_dem = current_DEM, input_shapefile = desired_shapefile, output_directory_images = image_patch_outputs_folder,output_directory_shapepatches = shapefile_patch_outputs,region_name= region, file_format = '.npy', numpy_output = DESIRED_NUMPY_OUTPUT, patch_size=DESIRED_PATCH_SIZE)
    
    print('patches creation done for' + region + '...')

    # Once region is done, delete the multibands and mosaics
    os.system('rm -rf ' + multiband_region_folder) 
    os.system('rm -rf ' + mosaics_outputs_folder + '/' + region)

    print('deleting multibands and mosaics done for' + region + '...')
    print('==============================================')

    ############# Preparing labels from shapefiles
    # We iterate through all shapefiles to generate yolo labels as .txt files
    # Obtain list of shapefiles in the shapefile_patch_outputs folder corresponding to this region
    shapefile_patch_list = [bandname for bandname in os.listdir(shapefile_patch_outputs) if bandname.startswith(region) and bandname.lower().endswith('.shp')]
    
    for shapefile_patch in shapefile_patch_list:
        shapefile_patch_path =  shapefile_patch_outputs + '/' + shapefile_patch
        labelfile_desired_path = labels_yolo_path + '/' + shapefile_patch[:-4] + '.txt'
        data_utils.generate_labels(shapefile_path = shapefile_patch_path,  target_label_path = labelfile_desired_path)

