###### This is to be run on ArcGIS Pro ######

# Forming Composite Bands

# Running a script to check all folders:
import os

def list_subfolders(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]

directory_path = r"D:\AiEarth\Gorkha landslide Roback\LandsatImages"
subfolders = list_subfolders(directory_path)

print(subfolders) # Shows desired list of subfolders

desired_list = ['LE07_L2SP_140041_20151008_20200903_02_T1_SR', 'LE07_L2SP_141040_20150609_20200904_02_T1_SR', 'LE07_L2SP_141041_20150524_20200904_02_T1_SR', 'LE07_L2SP_142040_20150531_20200904_02_T1_SR', 'LE07_L2SP_142041_20150531_20200904_02_T1_SR']
for folder in desired_list:
    # Setting the folder as our workspace
    arcpy.env.workspace = directory_path + "\\" + folder
    print(arcpy.env.workspace)
    string_r = folder + '_B4.tif'
    string_g = folder + '_B3.tif'
    string_b = folder + '_B2.tif'
    composite_string = string_r + ';' + string_g + ';' + string_b
    target_string = folder + '_BC.tif'
    arcpy.CompositeBands_management(composite_string,              target_string)

# Forming raster mosaic

arcpy.management.MosaicToNewRaster(
    input_rasters=r"'D:\AiEarth\Gorkha landslide Roback\LandsatImages\LE07_L2SP_140041_20151008_20200903_02_T1_SR\LE07_L2SP_140041_20151008_20200903_02_T1_SR_BC.tif';'D:\AiEarth\Gorkha landslide Roback\LandsatImages\LE07_L2SP_141040_20150609_20200904_02_T1_SR\LE07_L2SP_141040_20150609_20200904_02_T1_SR_BC.tif';'D:\AiEarth\Gorkha landslide Roback\LandsatImages\LE07_L2SP_141041_20150524_20200904_02_T1_SR\LE07_L2SP_141041_20150524_20200904_02_T1_SR_BC.tif';'D:\AiEarth\Gorkha landslide Roback\LandsatImages\LE07_L2SP_142040_20150531_20200904_02_T1_SR\LE07_L2SP_142040_20150531_20200904_02_T1_SR_BC.tif';'D:\AiEarth\Gorkha landslide Roback\LandsatImages\LE07_L2SP_142041_20150531_20200904_02_T1_SR\LE07_L2SP_142041_20150531_20200904_02_T1_SR_BC.tif'",
    output_location=r"D:\AiEarth\Gorkha landslide Roback\LandsatImages\Mosaic",
    raster_dataset_name_with_extension="LE07_L2SP_2015_02_T1_SR_BC_M.tif",
    coordinate_system_for_the_raster='PROJCS["WGS_1984_UTM_Zone_45N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",87.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]',
    pixel_type="16_BIT_UNSIGNED",
    cellsize=None,
    number_of_bands=3,
    mosaic_method="BLEND",
    mosaic_colormap_mode="FIRST"
)

# Breaking up multipolygon

# Geometry repair ensures that tool works correctly

arcpy.management.RepairGeometry(
    in_features="full_edit_final",
    delete_null="DELETE_NULL",
    validation_method="ESRI"
)

# Multipart to singlepart

arcpy.management.MultipartToSinglepart(
    in_features="full_edit_final",
    out_feature_class=r"D:\AiEarth\GorkhaEarthquake\GorkhaEarthquake.gdb\full_edit_final_multitosingle"
)

# Parent data folder:
p_data_folder = r"D:\AiEarth\Gorkha landslide Roback\LandsatImages\Mosaic"

#Output Images Folder
output_images_folder = r"D:\AiEarth\Gorkha landslide Roback\LandsatImages\Mosaic\image_patches"

#Clipped Output
clip_output_folder = r"D:\AiEarth\Gorkha landslide Roback\LandsatImages\Mosaic\shape_patches"

#TIF_clips
clip_tif_folder = r"D:\AiEarth\Gorkha landslide Roback\LandsatImages\Mosaic\tif_patches" 

# Set the current workspace
arcpy.env.workspace = p_data_folder

#Read Patches in workspace
rasters = arcpy.ListRasters()

for raster in rasters:
    print(raster)
    
patch_width = 640
patch_height = 640

shapefile_path = "full_edit_final_multitosingle"

# Name of the Mosaic File
mosaic_file = "LE07_L2SP_2015_02_T1_SR_BC_M.tif"

# Set the current workspace as the data folder that has the mosaic
arcpy.env.workspace = p_data_folder
    
# Split raster into patches

arcpy.management.SplitRaster(
    in_raster="LE07_L2SP_2015_02_T1_SR_BC_M.tif",
    out_folder=r"D:\AiEarth\Gorkha landslide Roback\LandsatImages\Mosaic\tif_patches",
    out_base_name="LE07_L2SP_2015_02_T1_SR_BC_M",
    split_method="SIZE_OF_TILE",
    format="TIFF",
    resampling_type="NEAREST",
    num_rasters="1 1",
    tile_size="640 640",
    overlap=0,
    units="PIXELS",
    cell_size=None,
    origin=None,
    split_polygon_feature_class=None,
    clip_type="NONE",
    template_extent="DEFAULT",
    nodata_value=""
)

#Set the workspace to the subfolder with .tif patches
arcpy.env.workspace = clip_tif_folder
    
#Read Patches in tif_patches folder relevant to current raster in progress
raster_patches = arcpy.ListRasters(raster.split('.')[0] + "*")
    
# For each raster patch, clip the shapefile
for raster_patch in raster_patches:

    #Clipping the shapefile
    clipped_shapefile_name = f"{raster_patch.split('.')[0]}.shp"
    clipped_shapefile_path = f"{clip_output_folder}\{clipped_shapefile_name}"
    #print(clipped_shapefile_path)

    temp_polygon = arcpy.ddd.RasterDomain(raster_patch,raster_patch+'raster_domain',"POLYGON")

    #arcpy.analysis.PairwiseClip(in_features = shapefile_path, clip_features = extent, out_feature_class = clipped_shapefile_path)
    arcpy.Clip_analysis(
        in_features=shapefile_path,
        clip_features=temp_polygon,
        out_feature_class=clipped_shapefile_path,
        cluster_tolerance=None)

    #Saving the patch as an image
    arcpy.conversion.RasterToOtherFormat(raster_patch,output_images_folder,"JPEG")