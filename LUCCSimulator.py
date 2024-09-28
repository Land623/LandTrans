import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.mps as mps
import rasterio
from utils.Metrics import  calculate_fom_for_tile

from Models.LandTrans_Tubelet import LandTrans_Tubelet,LandTrans_Tubelet_Decoder_Only
from Models.LandTrans_Patch import LandTrans_Patch,LandTrans_Patch_Decoder_Only
from Models.LandTrans_ST import LandTrans_ST,LandTrans_ST_Decoder_Only

class LUCCSimulator:
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

    def __load_checkpoint(self): 
        checkpoint = torch.load(self.model_file)
        self.embed_dim = checkpoint["embed_dim"]
        self.num_heads = checkpoint["num_heads"]
        self.num_encoder_layers = checkpoint["num_encoder_layers"]
        self.num_decoder_layers = checkpoint["num_decoder_layers"]
        self.num_land_use_types = checkpoint["num_land_use_types"]
        self.tubelet_size = checkpoint["tubelet_size"]
        self.batch_size = checkpoint["batch_size"]
        self.encoder_input_size = checkpoint["encoder_input_size"]
        self.decoder_input_size = checkpoint["decoder_input_size"]        
        self.label_len = checkpoint["label_len"]
        self.seq_len=checkpoint["seq_len"]
        self.sample_size=checkpoint["sample_size"]
        self.dropout = checkpoint["dropout"]
        self.dim_feedforward = checkpoint["dim_feedforward"]
        self.learning_rate = checkpoint["learning_rate"]
        self.patch_size=[self.tubelet_size[1],self.tubelet_size[2]]
        self.region_size=(self.encoder_input_size[1],self.encoder_input_size[2])
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
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
  
        self.model = self.model.to(self.device)  
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)   
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
 
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        return self.model, self.optimizer


    def Simulate(self, pred_loader,pred_sample_file_info,pred_len,
                                pred_output_folder):
        if not self.logger is None:
            self.logger.log_info("Loading model...")       
        self.__load_checkpoint()
        self.model.eval()
        file_name_list =  list(pred_sample_file_info.keys())
        file_count = 0   
        num_iterations = (pred_len + self.label_len - 1) // self.label_len     
        if num_iterations<1:
            num_iterations=1
        if not self.logger is None:            
            self.logger.log_info("Start simulation...")   
            self.logger.set_console(isVisible=False)
        with torch.no_grad():
            pred_bar = tqdm(pred_loader, desc="Simulating...")
            for inputs in pred_loader:
                inputs_enc = inputs.to(torch.long).to(self.device) 
                prediction = torch.zeros((inputs_enc.shape[0], pred_len, inputs_enc.shape[2],inputs_enc.shape[3]), device=self.device)
                for i in range(num_iterations):
                    if i == 0:
                        current_input_enc = inputs_enc
                        current_input_dec = inputs_enc[:, -self.label_len :, :, :] 
                    else:
                        current_input_enc = torch.cat([current_input_enc[:, self.label_len:, :,:], 
                                                prediction[:, (i-1)*self.label_len : i*self.label_len, :,:]], dim=1).to(torch.long)  
                        current_input_dec = current_input_enc[:, -self.label_len :, :, :] 
                    
                    current_prediction = self.model(current_input_enc,current_input_dec,return_logits=False, return_probs=False)      
                    
                    start_idx = i * self.label_len
                    end_idx = min((i + 1) * self.label_len, pred_len)
                    prediction[:, start_idx:end_idx,:, :] = current_prediction[:, :end_idx-start_idx, :,:]

                prediction = prediction[:, :pred_len,:, :]
                Batch_size = prediction.shape[0]              
                for b in range(Batch_size):       
                    cur_sample_file_name =  file_name_list[file_count]                     
                    cur_src =pred_sample_file_info[cur_sample_file_name] 
                    cur_src.update({'count': pred_len})
                    cur_pred_file_name = os.path.join(pred_output_folder,cur_sample_file_name)
                    cur_pred_file_name = cur_pred_file_name.replace('.tif','_pred.tif')   
                    cur_prediction = prediction[b:b+1,:,:,:].squeeze(0).cpu().numpy()  
                    with rasterio.open(cur_pred_file_name, "w", **cur_src) as dst:
                        dst.write(cur_prediction)    
                    file_count += 1
                pred_bar.update()
            pred_bar.close()    
        if not self.logger is None:
            self.logger.set_console(isVisible=True)
            self.logger.log_info("Simulation complete.") 