import os
import sys
import torch
import random
import numpy as np
import pandas as pd
import glob 
import json
import rasterio

import warnings
warnings.filterwarnings('ignore')

from utils.argparser import load_args_from_json,args_validation
from utils.log import Logger
from utils.DataEnvironment import DataEnvironment

from utils.log import Logger
from utils.Raster import generate_sample_areas,create_vrt_file_from_geotiff,delete_geotiff_and_auxiliaries,create_vrt_file_from_geotiff,export_vrt_bands
from utils.LanduseDataset import load_data
from LUCCTrainer import LUCCTrainer

from LUCCSimulator import LUCCSimulator

rasterio_path = os.path.dirname(rasterio.__file__)
proj_path = os.path.join(rasterio_path, 'proj_data')
os.environ['PROJ_LIB'] = proj_path

def clear_temporary_files(input_folder,output_vrt_file,merged_file_prefix):
    tile_files=create_vrt_file_from_geotiff(input_folder,output_vrt_file)
    export_vrt_bands(output_vrt_file, input_folder, merged_file_prefix)
    for file in tile_files:
        delete_geotiff_and_auxiliaries(file)
    os.remove(output_vrt_file)

def TrainModel(dataEnv,args,lucc_files, land_use_classes, logger,repeat_times):    
    best_fom=np.nan
    best_model_file=''    
    batch_size=args.batch_size  
    num_land_use_types=max(land_use_classes)+1
    embed_dim=args.embed_dim
    num_heads=args.num_heads
    learning_rate=args.learning_rate
    num_encoder_layers=args.num_encoder_layers
    num_decoder_layers=args.num_decoder_layers
    dropout=args.dropout
    dim_feedforward=args.dim_feedforward
    num_epochs=args.epochs
    patience=args.patience    
    seq_len=args.seq_len
    label_len=args.label_len
    pred_len=args.pred_len
    slide_interval=args.slide_interval
    sample_area_size=args.region_size
    patch_size=(sample_area_size,sample_area_size)
    spatial_patch_size=args.spatial_patch_size
    temporal_patch_size=args.temporal_patch_size
    tubelet_size=(temporal_patch_size,spatial_patch_size,spatial_patch_size)
    encoder_input_size=(seq_len,sample_area_size,sample_area_size)  
    decoder_input_size=(label_len,sample_area_size,sample_area_size) 
    model_type=args.model_type   
    periods=len(lucc_files)
    
    slide_window=seq_len+label_len
    test_time_point=[periods-slide_window]    
    val_time_point=periods-label_len-slide_window
    train_time_points=[]
    max_training_step=periods-label_len

    baseline_file=lucc_files[max_training_step-1]
    mask_file=""
    true_files=lucc_files[max_training_step:] 

    for t_step in range(max_training_step, slide_window-1,-slide_interval):      
        train_time_points.append(t_step-slide_window)
    if 0 not in train_time_points:
        train_time_points.append(0)
  
    config_info = {
        'seq_len':seq_len,
        'label_len':label_len,
        'pred_len':pred_len,
        'batch_size':batch_size,
        'tubelet_size':tubelet_size,
        'train_time_points':train_time_points,
        'val_time_points':val_time_point,
        'test_time_point':test_time_point,
        'slide_interval':slide_interval,
        'encoder_input_size':encoder_input_size,
        'decoder_input_size':decoder_input_size,
        'embed_dim':embed_dim,
        'num_heads':num_heads,
        'learning_rate':learning_rate,
        'num_encoder_layers':num_encoder_layers,
        'num_decoder_layers':num_decoder_layers,
        'dropout':dropout,
        'dim_feedforward':dim_feedforward,
        'num_epochs':num_epochs,
        'patience':patience,
        'model_type':model_type
    }    
    Samples_folders = dataEnv.SAMPLE_FOLDER
    masks_folder= dataEnv.MASK_FOLDER
    
    config_file = dataEnv.PARAMETER_FILE
    metric_csv_file=os.path.join(dataEnv.OUTPUT_FOLDER,"metrics.csv")
    with open(config_file, 'w') as f:
        json_str = json.dumps(config_info, indent=2)
        f.write(json_str)
    logger.log_info('Creating sample areas...')
    generate_sample_areas(lucc_files,patch_size,Samples_folders,masks_folder)

    lucc_sample_file_list = glob.glob(os.path.join(Samples_folders, '*.tiff')) + glob.glob(os.path.join(Samples_folders, '*.tif'))  
    random.shuffle(lucc_sample_file_list)
    logger.log_info('Loading training samples...')       
    train_loader,_ = load_data(lucc_sample_file_list, tubelet_size, seq_len, label_len,
            batch_size, train_time_points,sample_type=1)
    logger.log_info('Loading validating samples...') 
    val_loader,_ = load_data(lucc_sample_file_list, tubelet_size, seq_len, label_len,
            batch_size, train_time_points,sample_type=2)
    logger.log_info('Loading test samples...') 
    test_loader,test_sample_file_info = load_data(lucc_sample_file_list, tubelet_size, seq_len, label_len,
            batch_size, test_time_point,sample_type=3)
    
    metrics_df=None
    digt=len(str(repeat_times))
    for rt in range(repeat_times):
        treatmentID=str(rt+1).zfill(digt)
        dataEnv.check_treatment_folders(treatmentID)
        pred_output_folder= dataEnv.TEST_FOLDER
        acc_folder=dataEnv.ACCURACY_FOLDER         
        model_file=dataEnv.MODEL_FILE

        trainer=LUCCTrainer(model_file,model_type=model_type, logger=logger)   
        logger.log_info('Training...')
        logger.set_console(isVisible=False)    
        trainer.Train(train_loader,val_loader, num_epochs,patience,tubelet_size,batch_size, num_land_use_types,embed_dim,num_heads,
                            encoder_input_size,decoder_input_size,num_encoder_layers,num_decoder_layers,
                            learning_rate=learning_rate, dropout=dropout,dim_feedforward=dim_feedforward) 
        logger.set_console()
        logger.log_info('Testing...')
        '''cur_fom=trainer.Test(test_loader,test_sample_file_info,label_len, pred_output_folder,baseline_output_folder,
                    masks_folder,land_use_classes, acc_folder,acc_shp_file)'''    
        cur_fom,cur_overall_metrics_df = trainer.Test(test_loader,test_sample_file_info,pred_output_folder,label_len,
         baseline_file, mask_file, true_files,land_use_classes, acc_folder) 
        cur_overall_metrics_df['Run']='R'+treatmentID
        if metrics_df is None:
            metrics_df=cur_overall_metrics_df
        else:
            metrics_df=pd.concat([metrics_df,cur_overall_metrics_df])
        
        if np.isnan(best_fom):
            best_fom=cur_fom
            best_model_file=model_file
        elif best_fom<cur_fom:
            best_fom=cur_fom
            best_model_file=model_file             
        logger.set_console()     
    metrics_df.to_csv(metric_csv_file) 
    logger.log_info('Training Finished')
    return best_model_file

def Run_Train_Task(dataEnv,args,logger,repeat_times):   
    logger.log_info(f"Validation complete: {vali_result['number_of_lucc_data_files']}  lucc files have been validated.")   
    lucc_file_info=dataEnv.VALID_LUCC_INFO    
    lucc_file_list=list(lucc_file_info.keys())
    land_use_classes = vali_result['land_use_class']        
    best_model_file =  TrainModel(dataEnv,args,lucc_file_list, land_use_classes, logger,repeat_times)
    return best_model_file


def Simulation(dataEnv,args, logger,lucc_files,model_file):        
    batch_size=args.batch_size      
    seq_len=args.seq_len
    label_len=args.label_len
    pred_len=args.pred_len
    spatial_window_size=args.spatial_patch_size
    temporal_window_size=args.spatial_patch_size
    tubelet_size=(temporal_window_size,spatial_window_size,spatial_window_size)  
    model_type=args.model_type      
    Samples_folders = dataEnv.SAMPLE_FOLDER
    pred_folder=dataEnv.PREDICTION_FOLDER
    logger.log_info("Trained model used for simulation ["+model_file+"]")
    lucc_sample_file_list = glob.glob(os.path.join(Samples_folders, '*.tiff')) + glob.glob(os.path.join(Samples_folders, '*.tif'))  
    random.shuffle(lucc_sample_file_list)
    logger.log_info('Loading data...')    
    periods=len(lucc_files)   
    time_points=[periods-seq_len]
    pred_loader,pred_sample_file_info = load_data(lucc_sample_file_list, tubelet_size, seq_len, label_len,
            batch_size, time_points,sample_type=4)    
    dataEnv.check_folder(pred_folder)
    simulator=LUCCSimulator(model_file,model_type=model_type, logger=logger)
    simulator.Simulate(pred_loader,pred_sample_file_info,pred_len,pred_folder)   
    logger.log_info('Cleaning up temporary files...')
    cur_vrt_file_name=os.path.join(pred_folder,"Merged.vrt")
    clear_temporary_files(pred_folder,cur_vrt_file_name,"Pred")         
    logger.log_info('Finished')

def Run_Simulation_Task(dataEnv,args, logger, model_file,clear_temporary_files=False):         
    lucc_file_info=dataEnv.VALID_LUCC_INFO    
    lucc_file_list=list(lucc_file_info.keys())  
    Simulation(dataEnv,args, logger,lucc_file_list,model_file)   
    if clear_temporary_files:
        dataEnv.clear_temporary_files()

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)   
    repeat_times=1

    output_path = './Demo/Output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    config_files=['./Demo/Cfgs/CLCD_WH_Model12.json',
                  './Demo/Cfgs/NCLD_CLT_Model11.json']    
 
    for cfg_file in config_files:
        print(f"Running model with configuration file: {cfg_file}")    
        args = load_args_from_json(cfg_file)     
        vali_result =  args_validation(args)
        if vali_result['status'] != 'success':
            print(vali_result['message'])
            sys.exit()
        dataEnv=DataEnvironment(args,vali_result['lucc_file_info'])        
        odm_chk_result=dataEnv.check_folders()
        if odm_chk_result['status'] != 'success':
            print(odm_chk_result['message'])  
            sys.exit()
        log_file=dataEnv.LOG_FILE
        logger =Logger(log_file)         
        best_model_file = Run_Train_Task(dataEnv,args,logger, repeat_times) 
        if best_model_file!='' and os.path.isfile(best_model_file):
            Run_Simulation_Task(dataEnv,args,logger,best_model_file,clear_temporary_files=True)
        logger.exit()