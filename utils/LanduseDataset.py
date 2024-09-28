import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import rasterio
from copy import deepcopy
import os

class LandUseDataset(Dataset):
    def __init__(self, data, seq_len, label_len, sampling_point, sample_type=1):
        self.sample_type = sample_type  #1-training, 2-validation,3-test, 4-prediction 
        self.data = data
        self.window_size = seq_len + label_len
        if sample_type==4:
           self.window_size = seq_len
        self.encoder_input_size = seq_len
        self.decoder_input_size = label_len    
        self.sampling_point = sampling_point
       
        self.samples = []       
        self._create_samples()

    def _create_samples(self):
        file_idx=0
        S,T,H,W=self.data.shape
        lst_idx=self.sampling_point[-1]
        for land_use_t in self.data: 
            file_idx+=1
            for time_point_idx in self.sampling_point:   
                if time_point_idx + self.window_size <= land_use_t.shape[0]:
                    sampling=False
                    if self.sample_type ==3 or self.sample_type==4:
                        sampling=True  
                    elif self.sample_type ==1: 
                        if time_point_idx< lst_idx:
                            sampling=True  
                        else:
                            if float(file_idx/S)<=0.9:
                                sampling=True 
                    elif self.sample_type ==2: 
                        if time_point_idx== lst_idx and float(file_idx/S)>0.9:
                            sampling=True                   
                    if sampling:
                        input_seq = land_use_t[time_point_idx:time_point_idx+self.encoder_input_size]
                        if self.sample_type ==4:                            
                            self.samples.append((input_seq))
                        else:
                            target_seq = land_use_t[time_point_idx+self.encoder_input_size:time_point_idx+self.window_size]                        
                            self.samples.append((input_seq, target_seq))                
                            
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.sample_type ==4:  
            input_seq = self.samples[idx]
            return input_seq.long()
        else:   
            input_seq, label_seq = self.samples[idx]
            return input_seq.long(), label_seq.long()



def load_data(lucc_file_list, tube, seq_len, label_len,
            batch_size, Sampling_time_points, sample_type=1):    
    lucc_data = []
    sample_file_info={} 
    for lucc_file in lucc_file_list:
        with rasterio.open(lucc_file) as src:
            data = src.read()
            lucc_data.append(data)        
            sample_file_info[os.path.basename(lucc_file)]=deepcopy(src.profile)
    base_line_data = lucc_data[0] 
    N = len(lucc_data)
    T, H, W = base_line_data.shape
    t, h, w = tube 
       
    pad_h = (h - H % h) % h
    pad_w = (w - W % w) % w  
    data_tensor = torch.tensor(lucc_data)    
    if pad_w > 0 or pad_h > 0:
        padded_input = F.pad(data_tensor, (0, pad_w, 0, pad_h)) 
    else:
        padded_input = data_tensor

    ds = LandUseDataset(padded_input, seq_len, label_len, Sampling_time_points,sample_type)      

    data_loader = DataLoader(ds, batch_size = batch_size)
    return  data_loader,sample_file_info

def save_base_tgt(test_loader,Batch_size,test_sample_file_info,
                   label_len,tgt_folder,baseline_folder):
    file_name_list =  list(test_sample_file_info.keys())
    file_count = 0 
    for i, (inputs, targets) in enumerate(test_loader):
        for b in range(Batch_size):       
            baseline_data = inputs[b:b+1,-1:,:,:].squeeze(0)  
            tgt_data = targets[b:b+1,:,:,:].squeeze(0)

            cur_sample_file_name =  file_name_list[file_count]                     
            cur_src =test_sample_file_info[cur_sample_file_name] 
            cur_baseline_file_name = os.path.join(baseline_folder,cur_sample_file_name)
            cur_baseline_file_name = cur_baseline_file_name.replace('.tif','_base.tif')                
            cur_target_file_name = os.path.join(tgt_folder,cur_sample_file_name)
            cur_target_file_name = cur_target_file_name.replace('.tif','_tgt.tif')   
            cur_src.update({'count': label_len})
            with rasterio.open(cur_target_file_name, "w", **cur_src) as dst:
                dst.write(tgt_data)               
            cur_src.update({'count': 1})
            with rasterio.open(cur_baseline_file_name, "w", **cur_src) as dst:
                dst.write(baseline_data)
            file_count += 1
     
       
                
     

