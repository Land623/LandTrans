import os
import rasterio
import numpy as np
from sympy import true
import math
from affine import Affine

import glob
import os
from rasterio.merge import merge

from tqdm import tqdm
from osgeo import gdal

def create_vrt_file_from_geotiff(input_geotiff_folder,output_vrt_file_name):
    tiff_files = glob.glob(os.path.join(input_geotiff_folder, '*.tiff')) + glob.glob(os.path.join(input_geotiff_folder, '*.tif'))
    vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=False)
    gdal.BuildVRT(output_vrt_file_name, tiff_files, options=vrt_options)
    return tiff_files

def export_vrt_bands(vrt_path, output_folder, file_name,output_format="GTiff"):  
    dataset = gdal.Open(vrt_path)
    band_count = dataset.RasterCount 
    datatype = dataset.GetRasterBand(1).DataType
    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()
    options = [
    'COMPRESS=LZW',  
    'TILED=YES',     
    'BIGTIFF=YES',   
    'PREDICTOR=2'    
    ]
  
    output_files=[]
    digt=len(str(band_count))
    for band_num in range(1, band_count + 1):
        band = dataset.GetRasterBand(band_num)        
   
        output_file = os.path.join(output_folder, file_name +"_"+ str(band_num).zfill(digt)+".tif" )
        
     
        driver = gdal.GetDriverByName(output_format)
        output_dataset = driver.Create(output_file, dataset.RasterXSize, dataset.RasterYSize, 1, datatype,options=options)
        
   
        output_dataset.SetProjection(projection)
        output_dataset.SetGeoTransform(geotransform)
        
 
        output_dataset.GetRasterBand(1).WriteArray(band.ReadAsArray())
        

        output_dataset = None
        output_files.append(output_file)
    return output_files

def delete_geotiff_and_auxiliaries(file_path):
    if not os.path.exists(file_path):       
        return
    base_name = os.path.splitext(file_path)[0]
    aux_extensions = ['.tif', '.tif.aux.xml', '.tfw', '.ovr']
    for ext in aux_extensions:
        full_path = base_name + ext
        if os.path.exists(full_path):
            os.remove(full_path)

def ClipRaster(source_dir,target_dir,window):
    for filename in os.listdir(source_dir):
        if filename.endswith(".tif"):
            with rasterio.open(os.path.join(source_dir, filename)) as src:
                subset = src.read(window=window)
                print(filename)
                transform = rasterio.windows.transform(window, src.transform)
                profile = src.profile
                profile.update({                  
                    'height': window.height,
                    'width': window.width,
                    'transform': transform})
                with rasterio.open(os.path.join(target_dir, filename), 'w', **profile) as dst:
                    dst.write(subset)

def read_tif_info(tif_file_path):
    result = {'status': 'success', 'message': '', 'raster_properties':{}}
    try:
        with rasterio.open(tif_file_path) as dataset:
            data = dataset.read(1)
            unique_values = np.unique(data)          
            width = dataset.width
            height = dataset.height
            res = dataset.res
            count = dataset.count
            transform = dataset.transform
            raster_properties={
                'width': width,
                'height': height,
                'resolution': res,
                'band_count': count,
                'dtype':dataset.dtypes[0], 
                'geotransform': transform,
                'unique_values':unique_values
            }
            result['raster_properties']=raster_properties
            return result
    except Exception as e:
        result['status']='error'
        result['message']=f"Error reading {tif_file_path}: {e}"
        return result
    
def compare_geotiff_properties(raster_properties1, raster_properties2):
    result = {'status': 'success', 'message': '', 'match':False}
    try:
        tolerance=1.0
        resolution_match = (raster_properties1['resolution'] == raster_properties2['resolution'])        
        shape_match = (raster_properties1['width'] == raster_properties2['width']) and  (raster_properties1['height'] == raster_properties2['height'])        
        origin_match = (abs(raster_properties1['geotransform'][2] - raster_properties2['geotransform'][2]) <= tolerance and
                        abs(raster_properties1['geotransform'][5] - raster_properties2['geotransform'][5]) <= tolerance)
        if resolution_match and shape_match and origin_match :
            result['match']=True  
            result['status']='success'  
        return result    
    except Exception as e:
        result['status']='error'
        result['message']=f"Error comparing GeoTIFF properties: {e}"       
        return result
    
def read_file_info_from_list(data_folder, data_list_file):
    result = {'status': 'success', 'message': '', 'tif_info_dict' : {}}   
    tif_info_dict={}
    with open(data_list_file, 'r') as file:
        for line in file:
            tif_file_name = line.strip()
            if tif_file_name.lower().endswith('.tif'):
                tif_file_path = os.path.join(data_folder, tif_file_name)
                if os.path.exists(tif_file_path):
                    read_result=read_tif_info(tif_file_path)                
                    if read_result['status'] != 'success':
                        result['status']='error'
                        result['message']=read_result['message']
                        return result           
                    tif_info_dict[tif_file_path] = read_result['raster_properties']
                else:               
                    result['status']='error'
                    result['message']=f"File not found: {tif_file_path}"
                    return result     
        if result['status']=='success':
            result['tif_info_dict']=tif_info_dict
    return result

def validate_LUCC_files(data_folder, data_list_file):
    result = {'status': 'success', 'message': '', 'match' : False,'base_line_property':{}, 'number_of_files':0,'tif_info_dict':{}}   
    read_result= read_file_info_from_list(data_folder, data_list_file)
    if read_result['status'] != 'success':
        result['status']=read_result['status']
        result['message']=read_result['message']
        return result    
    tiff_info_dict=read_result['tif_info_dict']    
    if len(tiff_info_dict)==0:
        result['status']='error'
        result['message']='There is no validate geotiff file in the  directory you specified.'       
        return  result      
    baseline_key = next(iter(tiff_info_dict.keys()))    
    baseline_property=tiff_info_dict[baseline_key]
    result['base_line_property']=baseline_property
    if len(tiff_info_dict)==1:
        result['number_of_files']=1
        return result
    if baseline_property['band_count']>1:
        result['status']='error'
        result['message']=f'the land use/cover data {baseline_key} used for simulation should be a single band geo-tiff file.'
        return result
    if not np.issubdtype(baseline_property['dtype'] , np.integer):
        result['status']='error'
        result['message']=f'The land use/cover data {baseline_key} used for simulation must be of integer data type.'
        return result   
    for key in tiff_info_dict.keys():
        if key != baseline_key:          
            cur_property=tiff_info_dict[key]   
            if cur_property['band_count']>1:
                result['status']='error'
                result['message']=f'the land use/cover data {key} used for simulation should be a single band geo-tiff file.'
                return result  
            if not np.issubdtype(cur_property['dtype'] , np.integer):
                result['status']='error'
                result['message']=f'The land use/cover data {key} used for simulation must be of integer data type.'
                return result         
            cmp_result = compare_geotiff_properties(baseline_property, cur_property)  
            if  cmp_result['status'] != 'success':
                result['status']=cmp_result['status']
                result['message']=cmp_result['message']
                return result
            if cmp_result['match']==False: 
                result['status']='error'              
                result['match']=False
                result['message']=f'The extent of {key} does not match the base line land use/cover file.'
                return result    
    result['number_of_files']=len(tiff_info_dict)
    result['tif_info_dict']=tiff_info_dict
    return result

def combine_multi_tiffs(input_geotiff_files,output_image_file):
    # Open the input files
    datasets = [rasterio.open(file) for file in input_geotiff_files]

    # Check if all files have the same size
    for i in range(1, len(datasets)):
        if datasets[0].shape != datasets[i].shape:
            raise ValueError("Files have different sizes")

    # Create the output file
    with rasterio.open(output_image_file, 'w', 
                        driver='GTiff', 
                        height=datasets[0].shape[0], 
                        width=datasets[0].shape[1], 
                        count=len(datasets), 
                        dtype=np.uint8, 
                        crs=datasets[0].crs, 
                        nodata=datasets[0].nodata,
                        transform=datasets[0].transform) as dst:
        for i, dataset in enumerate(datasets):
            dst.write(dataset.read(1).astype(np.uint8), 1 + i)
    # Close the input files
    for dataset in datasets:
        dataset.close()


def generate_sample_areas(lucc_map_file_list,patch_size,output_folder, mask_folder):
    base_line_map_file=lucc_map_file_list[0]
    band_count=len(lucc_map_file_list)
    with rasterio.open(base_line_map_file) as src:
        profile = src.profile
        rows = profile['height']
        cols = profile['width']           
        # Define the patch size
        h, w = patch_size
        num_H=(rows + h - 1) // h
        num_W=(cols + w - 1) // w         
        num_digits_row = int(math.log10(abs(num_H))) + 1
        num_digits_col = int(math.log10(abs(num_W))) + 1
        region_count=1
        
        total_mask_file = os.path.join(mask_folder, 'mask.tif')
        baseline_data = src.read(1)        
        total_mask_area = np.where(baseline_data != src.nodata, 1, 0)        
        with rasterio.open(total_mask_file, 'w', **profile) as dst:
            dst.write(total_mask_area,1)     

        # Iterate over the patches
        Row_bar = tqdm(range(num_H),'Row') 
        Col_bar = tqdm(range(num_W),'Col')       
        for i in range(num_H):          
            Col_bar.reset()
            Col_bar.refresh()          
            for j in range(num_W): 
                # Calculate the patch coordinates  
                patch_y1 = i * h
                patch_y2 = (i + 1) * h
                patch_x1 = j * w
                patch_x2 = (j + 1) * w
                window = rasterio.windows.Window(patch_x1, patch_y1, w, h)
                transform = rasterio.windows.transform(window, src.transform)
                # Pad the patch if necessary
                if patch_y2 > rows:
                    window = rasterio.windows.Window(patch_x1, patch_y1, w, rows-patch_y1-1) 
                    transform = rasterio.windows.transform(window, src.transform)                       
                if patch_x2 > cols:
                    window = rasterio.windows.Window(patch_x1, patch_y1, cols-patch_x1-1, h)   
                    transform = rasterio.windows.transform(window, src.transform)                           
                # Extract the patch 
                patch_data_list=[]   
                patch_base_line = src.read(window=window).astype(np.uint8)
                patch_data_list.append(patch_base_line)
                for file_idx in range(1,len(lucc_map_file_list)):
                    with rasterio.open(lucc_map_file_list[file_idx]) as cur_src:
                        cur_patch = cur_src.read(window=window).astype(np.uint8)
                        patch_data_list.append(cur_patch)               
                patch_array = np.array(patch_data_list) 
                patch = np.squeeze(patch_array, axis=1)  
                padded=False
                if patch_y2 > rows or patch_x2 > cols:   
                    pad_height = h- patch.shape[1]
                    pad_width =  w- patch.shape[2]    
                    # Calculate the padding for height and width
                    pad_height_before = pad_height // 2
                    pad_height_after = pad_height - pad_height_before
                    pad_width_before = pad_width // 2
                    pad_width_after = pad_width - pad_width_before              
                    transform = Affine( transform.a, transform.b, transform.c- transform.a* pad_width_before,
                           transform.d, transform.e , transform.f + transform.a*pad_height_before)  
                    # Apply padding
                    patch = np.pad(patch, ((0, 0), (pad_height_before, pad_height_after), (pad_width_before, pad_width_after)), mode='constant')
                    padded=True    
                # Check if the patch contains only nodata values
                mask_area = np.where(patch[0] != src.nodata, 1, 0)
                if np.sum(mask_area)!=0:                   
                    # Create a new profile for the output file
                    profile.update({
                        'height': patch.shape[1],
                        'width': patch.shape[2], 
                        'count':band_count,
                        'transform': transform,
                        'crs': src.crs,
                        'dtype': rasterio.uint8
                    })
                    out_put_file=os.path.join(output_folder,'Region_R'+str(i).zfill(num_digits_row)+'_C'+str(j).zfill(num_digits_col)+'.tif')
                    # Write the patch to a new file                  
                    with rasterio.open(out_put_file, 'w', **profile) as dst:
                        dst.write(patch)   

                    mask_file=os.path.join(mask_folder,'Region_R'+str(i).zfill(num_digits_row)+'_C'+str(j).zfill(num_digits_col)+'_mask.tif')                 
                    profile.update({
                            'count':1,
                        })                   
                    with rasterio.open(mask_file, 'w', **profile) as dst:
                        dst.write(mask_area,1) 
                    region_count=region_count+1    
                Col_bar.update(1) 
            Row_bar.update()   
        Col_bar.close()
        Row_bar.close()        
        return region_count



def merge_geotiff_files(input_dir,output_file):   
    tif_files = glob.glob(os.path.join(input_dir, '*.tif'))
    merge_geotiff_files_from_list(tif_files,output_file)

    
        
def merge_geotiff_files_from_list(file_list,output_file):   
    src_files_to_mosaic = []
    for fp in file_list:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files_to_mosaic:
        src.close()
   
def split_multi_band_geotff(input_file,output_dir):   
    os.makedirs(output_dir, exist_ok=True)
    base_file_name=os.path.basename(input_file)

    with rasterio.open(input_file) as src:
        for band_id in range(1, src.count + 1):
            band_data = src.read(band_id)          

            out_meta = src.meta.copy()
            out_meta.update({
                "count": 1,
                "dtype": band_data.dtype
            })
            
            output_file = os.path.join(output_dir, base_file_name )
            output_file = output_file.replace('.tif',f'_band_{band_id}.tif')           

            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(band_data, 1)

