import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import torch.backends.mps as mps
import rasterio
import geopandas as gpd
from shapely.geometry import box

from sklearn.metrics import  confusion_matrix
from utils.Metrics import  calculate_fom_for_tile, calculate_fom,calculate_metrics_from_conf_matrix
from utils.Raster import create_vrt_file_from_geotiff,export_vrt_bands,delete_geotiff_and_auxiliaries

from Models.LandTrans_Tubelet import LandTrans_Tubelet,LandTrans_Tubelet_Decoder_Only
from Models.LandTrans_Patch import LandTrans_Patch,LandTrans_Patch_Decoder_Only
from Models.LandTrans_ST import LandTrans_ST,LandTrans_ST_Decoder_Only

class LUCCTrainer:
    def __init__(
        self,
        model_file,  
        model_type=0,      
        logger=None,
    ):
        self.Multi_GPU=False
        device_type = "cpu"
        if torch.cuda.is_available():
            device_type = "cuda"
            if torch.cuda.device_count() > 1:
               self.Multi_GPU=True              
        elif mps.is_available():
            device_type = "mps"      
        self.device = torch.device(device_type)  
        self.model_file = model_file        
        self.logger = logger
        self.model_type = model_type  

    def __save_checkpoint(self):
        checkpoint = {   
            "embed_dim":self.embed_dim ,
            "num_heads":self.num_heads,
            "num_encoder_layers":self.num_encoder_layers,
            "num_decoder_layers":self.num_decoder_layers,
            "num_land_use_types":self.num_land_use_types, 
            "tubelet_size":self.tubelet_size, 
            "batch_size":self.batch_size,
            "encoder_input_size":self.encoder_input_size,
            "decoder_input_size":self.decoder_input_size,             
            "dropout":self.dropout,
            "dim_feedforward":self.dim_feedforward,
            "learning_rate":self.learning_rate,                   
            "sample_size":self.sample_size,
            "label_len":self.label_len, 
            "seq_len":self.seq_len,    
            "model_type":self.model_type,   
            "region_size": self.region_size,     
            "optimizer_state_dict": self.optimizer.state_dict()                
        }
    
        if isinstance(self.model, torch.nn.DataParallel):
            checkpoint["model_state_dict"] = self.model.module.state_dict()
        else:
            checkpoint["model_state_dict"] = self.model.state_dict()
        torch.save(checkpoint, self.model_file)

    def Train(self, train_loader, valid_loader,num_epochs, patience, tubelet_size,
        batch_size, num_land_use_types, embed_dim, num_heads, input_size_encoder,
        input_size_decoder, num_encoder_layers, num_decoder_layers, learning_rate=0.0001,
        dropout=0.1, dim_feedforward=2048):  
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_land_use_types = num_land_use_types
        self.tubelet_size = tubelet_size
        self.batch_size = batch_size
        self.encoder_input_size = input_size_encoder
        self.decoder_input_size = input_size_decoder        
        self.label_len = input_size_decoder[0]
        self.seq_len = input_size_encoder[0]        
        self.sample_size=input_size_encoder[1]    
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.learning_rate = learning_rate     
        self.patch_size=[self.tubelet_size[1],self.tubelet_size[2]]   
        self.region_size=(input_size_decoder[1],input_size_decoder[2])
        if self.model_type=="M11":
            self.model = LandTrans_Patch(self.seq_len, self.label_len, 
                 self.patch_size,self.encoder_input_size,self.decoder_input_size,
                 self.num_land_use_types, self.embed_dim, 
                 self.num_heads, self.num_encoder_layers,self.num_decoder_layers, 
                 dropout=self.dropout, dim_feedforward=self.dim_feedforward) 
        elif self.model_type=="M12":
            self.model = LandTrans_Patch_Decoder_Only(self.seq_len, self.label_len,
                    self.patch_size, self.encoder_input_size,self.decoder_input_size,
                    self.num_land_use_types, self.embed_dim, 
                    self.num_heads,  self.num_decoder_layers,
                    dropout=self.dropout, dim_feedforward=self.dim_feedforward)   
        elif self.model_type=="M21":
            self.model = LandTrans_Tubelet(self.seq_len, self.label_len, 
                    self.encoder_input_size,self.decoder_input_size,
                    self.tubelet_size, self.num_land_use_types, self.embed_dim, 
                    self.num_heads, self.num_encoder_layers, self.num_decoder_layers,
                    dropout=self.dropout, dim_feedforward=self.dim_feedforward)   
        elif self.model_type=="M22":
            self.model = LandTrans_Tubelet_Decoder_Only(self.seq_len, self.label_len,
                    self.encoder_input_size,self.decoder_input_size,
                    self.tubelet_size, self.num_land_use_types, self.embed_dim, 
                    self.num_heads, self.num_decoder_layers,
                    dropout=self.dropout, dim_feedforward=self.dim_feedforward)            
        elif self.model_type=="M31":        
            self.model = LandTrans_ST(self.seq_len, self.label_len, self.region_size,
                    self.patch_size,  self.num_land_use_types, self.embed_dim, 
                    self.num_heads, self.num_heads, self.num_encoder_layers, self.num_decoder_layers,
                    dropout=self.dropout, dim_feedforward=self.dim_feedforward)
        elif self.model_type=="M32":        
            self.model = LandTrans_ST_Decoder_Only(self.seq_len, self.label_len,self.region_size,
                    self.patch_size,  self.num_land_use_types, self.embed_dim, 
                    self.num_heads, self.num_heads, self.num_encoder_layers, self.num_decoder_layers,
                    dropout=self.dropout, dim_feedforward=self.dim_feedforward)        
        if self.Multi_GPU:
            self.model = nn.DataParallel(self.model)   
        self.model=self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        best_val_loss = np.inf
        counter = 0        
        epochs_bar = tqdm(range(self.num_epochs), desc="Epochs")
        batches_bar = tqdm(total=len(self.train_loader), desc="Batches")
        for epoch in epochs_bar:
            running_loss = 0.0
            self.model.train()
            for i, (inputs_enc, targets) in enumerate(train_loader):                
                self.optimizer.zero_grad()          
                inputs_enc = inputs_enc.to(torch.long).to(self.device)
                inputs_dec = inputs_enc[:, -self.label_len :, :, :] 
                targets = targets.to(torch.long).to(self.device)  
                outputs = self.model(inputs_enc,inputs_dec, return_logits=True) 
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                batches_bar.update()
            avg_loss = running_loss / len(self.train_loader)
            valid_loss = self.__validate()
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                self.__save_checkpoint()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    if self.logger is not None:
                        self.logger.log_warning("Early stopping!")
                    break
            if self.logger is not None:
                self.logger.log_info(
                    f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Valid loss: {valid_loss:.4f}"
                )
            epochs_bar.set_postfix(
                {
                    "Epoch": epoch + 1,
                    "Average Loss": round(avg_loss, 4),
                    "Valid loss": round(valid_loss, 4),
                }
            )
            batches_bar.refresh()
            batches_bar.reset()
        batches_bar.close()
        epochs_bar.close()  

    def __validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs_enc, targets in self.valid_loader:
                inputs_enc = inputs_enc.to(torch.long).to(self.device)
                inputs_dec = inputs_enc[:, -self.label_len :, :, :]
                targets = targets.to(torch.long).to(self.device)  
                outputs = self.model(inputs_enc,inputs_dec, return_logits=True)   
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(self.valid_loader)

    def Test0(self, test_loader,test_sample_file_info,label_len,
                                pred_output_folder,baseline_output_folder,
                                masks_folder,land_use_classes, acc_folder,acc_shp_file):
        self.model.eval()
        file_name_list =  list(test_sample_file_info.keys())
        file_count = 0    
        Fom=[]
        A=[]
        B=[]
        C=[]
        D=[]
        maxtric_list=[] 
        for ts in range(label_len+1):
            Fom.append(0)
            A.append(0)
            B.append(0)
            C.append(0)
            D.append(0)
     
        acc_info_list = {}
        overall_accuracy_list =[]
        Region_crs = None
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc="Test")
            for i, (inputs, targets) in enumerate(test_loader):
                inputs_enc = inputs.to(torch.long).to(self.device) 
                inputs_dec = inputs_enc[:, -self.label_len :, :, :] 
                targets = targets.to(torch.long).to(self.device)                
                outputs = self.model(inputs_enc,inputs_dec,return_logits=False, return_probs=False)            
                Batch_size = outputs.shape[0]
                K = outputs.shape[1]                
                for b in range(Batch_size):       
                    cur_sample_file_name =  file_name_list[file_count]                     
                    cur_src =test_sample_file_info[cur_sample_file_name] 
                    cur_src.update({'count': self.label_len})
                    cur_baseline_file_name = os.path.join(baseline_output_folder,cur_sample_file_name)
                    cur_baseline_file_name = cur_baseline_file_name.replace('.tif','_base.tif')   
                    cur_mask_file_name = os.path.join(masks_folder,cur_sample_file_name)  
                    cur_mask_file_name = cur_mask_file_name.replace('.tif','_mask.tif')  
                    cur_pred_file_name = os.path.join(pred_output_folder,cur_sample_file_name)
                    cur_pred_file_name = cur_pred_file_name.replace('.tif','_pred.tif')                                   

                    with rasterio.open(cur_mask_file_name) as dst:
                        mask_area = dst.read(1)
                        bounds = dst.bounds
                        geo = box(*bounds)   
                        if Region_crs is None:
                            Region_crs=dst.crs   
                  
                    cur_baseline = inputs[b:b+1,-1:,:,:].squeeze(0)   
                    cur_prediction = outputs[b:b+1,:,:,:].squeeze(0).cpu().numpy()                    
                    cur_target = targets[b:b+1,:,:,:].squeeze(0).cpu().numpy()  
                    with rasterio.open(cur_pred_file_name, "w", **cur_src) as dst:
                        dst.write(cur_prediction)                    
                    cur_tile_id=  cur_sample_file_name.replace('.tif','')   
                    cur_acc_info = [cur_tile_id, geo]    
                    sub_Fom, sub_A, sub_B, sub_C, sub_D, sub_accuracy_list, sub_maxtric_list, _, _ = calculate_fom_for_tile(
                    cur_baseline,
                    cur_target,
                    cur_prediction,
                    land_use_classes,
                    mask=mask_area)
                    cur_acc_info.extend(sub_Fom)
                    cur_acc_info.extend(sub_A)
                    cur_acc_info.extend(sub_B)
                    cur_acc_info.extend(sub_C)
                    cur_acc_info.extend(sub_D)
                    cur_acc_info.extend(sub_accuracy_list)
                    for t in range(label_len+1):                    
                        A[t]=A[t] + sub_A[t]
                        B[t]=B[t] + sub_B[t]
                        C[t]=C[t] + sub_C[t]
                        D[t]=D[t] + sub_D[t]                                            
                        if file_count==0:
                            maxtric_list.append(sub_maxtric_list[t])
                        else:
                            maxtric_list[t] = maxtric_list[t] + sub_maxtric_list[t]   
                    acc_info_list[cur_tile_id] = cur_acc_info    
                    file_count += 1
                test_bar.update()
            test_bar.close()
        column_names = ["Name", "geometry"]
        metrics_list = ["Fom", "A", "B", "C", "D", "OA"]  
        for mc in metrics_list:
            for c in range(label_len + 1):
                column_names.append(mc +  "_T" + str(c))
        total_acc = gpd.GeoDataFrame(list(acc_info_list.values()), columns=column_names)
        total_acc.crs = Region_crs
        total_acc.to_file(acc_shp_file)
        for ts in range(label_len+1):
            Fom[ts]=round(B[ts]/(A[ts]+B[ts]+C[ts]+D[ts]),6)
            cur_matrix_file = os.path.join(
                        acc_folder,
                        "ConfusionMatrix_T"+ str(ts)+ ".csv")
            conf_matrix= maxtric_list[ts].iloc[:, 3:].values  
            precision =np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
            recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
            f1 = 2 * precision * recall / (precision + recall)     
            maxtric_list[ts]['Precision']=precision
            maxtric_list[ts]['Recall']=recall
            maxtric_list[ts]['F1 Score']=f1         
            overall_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
            overall_accuracy=round(overall_accuracy,6)
            overall_accuracy_list.append(overall_accuracy)  
            maxtric_list[ts].to_csv(cur_matrix_file)
        if self.logger is not None:
            self.logger.log_info("Fom "+str(Fom))
            self.logger.log_info("OA "+str(overall_accuracy_list))
        else:
            print(Fom)
            print(overall_accuracy_list)
        return Fom[0]

    def Test(self, test_loader,test_sample_file_info,pred_output_folder,label_len,
            baseline_file, mask_file, true_files,land_use_classes, acc_folder):
            self.model.eval()
            file_name_list =  list(test_sample_file_info.keys())
            file_count = 0  

            A_list = np.zeros(label_len+1)
            B_list = np.zeros(label_len+1)
            C_list = np.zeros(label_len+1)
            D_list = np.zeros(label_len+1)
            Fom=np.zeros(label_len+1)
            OA_List = np.zeros(label_len+1)
            Kappa = np.zeros(label_len+1)
            combined_confusion_matrix=None   
            total_precision = np.zeros(label_len+1)
            total_recall = np.zeros(label_len+1)
            total_f1 = np.zeros(label_len+1)    
            num_digits = len(str(label_len))  

            string_list = ['overall'] + [f'T{i+1:0{num_digits}d}' for i in range(label_len)]
            with torch.no_grad():
                test_bar = tqdm(test_loader, desc="Test")
                for i, (inputs, targets) in enumerate(test_loader):
                    inputs_enc = inputs.to(torch.long).to(self.device) 
                    inputs_dec = inputs_enc[:, -self.label_len :, :, :] 
                    targets = targets.to(torch.long).to(self.device)                
                    outputs = self.model(inputs_enc,inputs_dec,return_logits=False, return_probs=False)            
                    Batch_size = outputs.shape[0]
                    K = outputs.shape[1]                
                    for b in range(Batch_size):       
                        cur_sample_file_name =  file_name_list[file_count]                     
                        cur_src =test_sample_file_info[cur_sample_file_name] 
                        cur_src.update({'count': self.label_len})                    
                        cur_pred_file_name = os.path.join(pred_output_folder,cur_sample_file_name)
                        cur_pred_file_name = cur_pred_file_name.replace('.tif','_pred.tif')    
                        cur_prediction = outputs[b:b+1,:,:,:].squeeze(0).cpu().numpy()  
                        with rasterio.open(cur_pred_file_name, "w", **cur_src) as dst:
                            dst.write(cur_prediction)   
                            if not dst.closed:
                                dst.close()
                        file_count += 1
                    test_bar.update()
                test_bar.close()
            if self.logger is not None:                
                self.logger.log_info("Calculating precision metrics, please wait...")              
            else:
                print("Calculating precision metrics, please wait...")
            
            output_merged_vrt_file=os.path.join(pred_output_folder,"Merged.vrt")
            tile_files=create_vrt_file_from_geotiff(pred_output_folder,output_merged_vrt_file)
            output_single_band_files = export_vrt_bands(output_merged_vrt_file, pred_output_folder, "Test")            
            for file in tile_files:
                delete_geotiff_and_auxiliaries(file)
            os.remove(output_merged_vrt_file)

            mask=None
            with rasterio.open(baseline_file) as baseline_src:
                baseline_data = baseline_src.read(1)  
            if mask_file!="" and os.path.exists(mask_file):
                with rasterio.open(mask_file) as mask_src:
                    mask_data = mask_src.read(1)    
                    mask = np.array(mask) 
                    mask = mask.astype(bool)    
            baseline_data_masked = baseline_data
            if mask is not None:
                baseline_data_masked=baseline_data[mask]

            file_count=0
            cal_bar = tqdm(total=label_len, desc="Calculating...")
            for cur_pred_file in output_single_band_files:
                true_file=true_files[file_count]
                with rasterio.open(true_file) as true_src:
                    true_data = true_src.read(1)  
                 
                with rasterio.open(cur_pred_file) as pred_src:
                    pred_data = pred_src.read(1)  

                true_end_t = true_data
                pred_end_t = pred_data
                if mask is not None:
                    true_end_t=true_data[mask]
                    pred_end_t=pred_data[mask]
          
                cur_Fom, cur_A, cur_B, cur_C, cur_D = calculate_fom(baseline_data_masked,true_end_t,pred_end_t)
         
                A_list[file_count+1] = cur_A  
                B_list[file_count+1] = cur_B    
                C_list[file_count+1] = cur_C  
                D_list[file_count+1] = cur_D  
        
                Fom[file_count+1] = cur_Fom   

                true_end_t=true_end_t.flatten() 
                pred_end_t=pred_end_t.flatten() 

                cm = confusion_matrix(true_end_t,pred_end_t,labels=land_use_classes)                
                if combined_confusion_matrix is None:
                    combined_confusion_matrix=cm
                else:
                    combined_confusion_matrix = combined_confusion_matrix+cm
                oa_matrix,cur_OA,cur_Kappa,precision,recall,F1_Score =calculate_metrics_from_conf_matrix(cm,land_use_classes)
                OA_List[file_count+1]=cur_OA
                Kappa[file_count+1]=cur_Kappa
                total_precision[file_count+1]=precision
                total_recall[file_count+1]=recall
                total_f1[file_count+1]= F1_Score

                cur_matrix_file = os.path.join(
                        acc_folder,
                        "ConfusionMatrix_T"+ str(file_count+1)+ ".csv")
                oa_matrix.to_csv(cur_matrix_file)
                file_count+=1      
                cal_bar.update()
            cal_bar.close()      
               
            A_list[0]=A_list.sum()
            B_list[0]=B_list.sum()
            C_list[0]=C_list.sum()
            D_list[0]=D_list.sum()
            sum_ABCD=A_list[0] +B_list[0] + C_list[0] + D_list[0]
            Fom[0]=B_list[0] / (sum_ABCD) if sum_ABCD != 0 else 0   
            Fom[0]= round(Fom[0],6)

            overall_matrix, overall_OA,overall_Kappa,overall_precision,overall_recall,overall_F1 =calculate_metrics_from_conf_matrix(combined_confusion_matrix,land_use_classes)
            overall_matrix_file = os.path.join(
                        acc_folder,
                        "ConfusionMatrix_Overall.csv")
            overall_matrix.to_csv(overall_matrix_file)
            OA_List[0]=overall_OA
            Kappa[0]=overall_Kappa
            total_precision[0]=overall_precision
            total_recall[0]=overall_recall
            total_f1[0]= overall_F1
            if self.logger is not None:
                self.logger.log_info("Fom "+str(Fom))
                self.logger.log_info("OA "+str(OA_List))
                self.logger.log_info("Kappa "+str(Kappa))
            else:
                print(Fom)
                print(OA_List)
                print(Kappa)  

            overall_metrics={
                'Step':string_list,
                'Figure of Merit':Fom,
                'Overall accuracy':OA_List,
                'Kappa':Kappa,
                'Precision':total_precision,
                'Recall':total_recall,
                'F1 Score':total_f1
            }
            overall_metrics_df=pd.DataFrame(overall_metrics)
            return Fom[0],overall_metrics_df
