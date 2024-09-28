import torch
import torch.nn as nn
import torch.nn.functional as F

class LUCCSpatialEmbedding(nn.Module):
    def __init__(self, num_classes, embed_dim, patch_size, region_size):
        super(LUCCSpatialEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_classes, embed_dim)   
        self.patch_size = patch_size  # (h, p)
        self.projection = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # Position embedding        
        num_patches = (region_size[0] // patch_size[0]) * (region_size[1] // patch_size[1])
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1, num_patches, embed_dim))  # (1, 1, num_patches, embed_dim)

    def forward(self, land_use_maps):  
        land_use_maps = self.embedding(land_use_maps)                 
        B, T, H, W, C = land_use_maps.shape
        land_use_maps = land_use_maps.view(B * T, H, W, C).permute(0, 3, 1, 2)  # [B*T, C, H, W]     
        # Apply Conv2d to get embeddings
        x = self.projection(land_use_maps)  # (B*T, embed_dim, H//H_p, W//W_p)    
        # Reshape to (B, T, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2).view(B, T, -1, self.embed_dim)
        # Add position embeddings (broadcasting)
        x = x + self.position_embeddings       
        return x

class LUCCTemporalEmbedding(nn.Module):
    def __init__(self, embed_dim, pred_len):
        super(LUCCTemporalEmbedding, self).__init__()
        self.embed_dim = embed_dim 
        self.TemporalPositionEmbedding = nn.Parameter(torch.zeros(1, pred_len, 1, embed_dim))  # (1, 1, num_patches, embed_dim)
    def forward(self, x):    
        return x + self.TemporalPositionEmbedding


class STEncoder(nn.Module):
    def __init__(self, embed_dim, spatial_heads, temporal_heads, spatial_layers, temporal_layers, dropout, dim_feedforward):
        super(STEncoder, self).__init__()
        self.spatial_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=spatial_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(spatial_layers)
        ])
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=temporal_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(temporal_layers)
        ])

    def forward(self, x):
        batch_size, num_frames, num_tokens, embed_dim = x.shape
        # Spatial attention
        x = x.view(batch_size * num_frames, num_tokens, embed_dim).permute(1, 0, 2)  # [num_tokens, batch_size * num_frames, embed_dim]
        for layer in self.spatial_layers:
            x = layer(x)
        x = x.permute(1, 0, 2).view(batch_size, num_frames, num_tokens, embed_dim)  # [batch_size, num_frames, num_tokens, embed_dim]         
        # Temporal attention         
        x = x.permute(0, 2, 1, 3)  # [batch_size, num_tokens, num_frames, embed_dim]         
        x = x.reshape(batch_size * num_tokens, num_frames, embed_dim).permute(1, 0, 2)   
        if num_frames>1:
            for layer in self.temporal_layers:
                x = layer(x)
        x = x.permute(1, 0, 2).reshape(batch_size, num_tokens, num_frames, embed_dim)  # [batch_size, num_tokens, num_frames, embed_dim]
        x = x.permute(0, 2, 1, 3)  # [batch_size, num_frames, num_tokens, embed_dim]
        return x

class STDecoder(nn.Module):
    def __init__(self, embed_dim, spatial_heads, temporal_heads, num_spatial_layers, num_temporal_layers, dropout, dim_feedforward):
        super(STDecoder, self).__init__()
        self.spatial_layers = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=spatial_heads, dropout=dropout, dim_feedforward=dim_feedforward, batch_first=True)
        self.spatial_decoder = nn.TransformerDecoder(self.spatial_layers, num_layers=num_spatial_layers)
        self.temporal_layers = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=temporal_heads, dropout=dropout, dim_feedforward=dim_feedforward, batch_first=True)
        self.temporal_decoder = nn.TransformerDecoder(self.temporal_layers, num_layers=num_temporal_layers)

    def forward(self, y, memory, tgt_mask=None):  
        B, num_frames_y, num_patches_y, embed_dim_y = y.shape
        B_m, num_frames_m, num_patches_m, embed_dim_m = memory.shape
        decoded_tokens = y.reshape(B,  num_frames_y* num_patches_y, embed_dim_y)  # [num_patches_y, B * num_frames_y, embed_dim_y]
        memory = memory.reshape(B,  num_frames_m* num_patches_m, embed_dim_m)  # [num_patches_m, B * num_frames_m, embed_dim_m]
        if tgt_mask is not None:
            decoded_tokens=self.spatial_decoder(decoded_tokens,memory,tgt_mask=tgt_mask) 
        else:
            decoded_tokens=self.spatial_decoder(decoded_tokens,memory) 
        if num_frames_y>1:
            if tgt_mask is not None:
                decoded_tokens=self.temporal_decoder(decoded_tokens,memory,tgt_mask=tgt_mask) 
            else:
                decoded_tokens=self.temporal_decoder(decoded_tokens,memory)               
        decoded_tokens = decoded_tokens.view(B, num_frames_y, num_patches_y, embed_dim_y)  # [B, num_frames_y, num_patches_y, embed_dim_y]
        return decoded_tokens


class LandTrans_ST(nn.Module):
    def __init__(self, seq_len, pred_len,region_size, 
                 patch_size, num_classes, embed_dim, 
                 spatial_heads, temporal_heads, spatial_layers, temporal_layers,         
                 dropout=0.1, dim_feedforward=2048):
        super(LandTrans_ST, self).__init__()      
        self.embed_dim = embed_dim
        self.seq_len = seq_len  # 输入序列长度
        self.pred_len = pred_len  # 目标序列长度
        self.num_classes = num_classes 
        self.patch_size = patch_size   
        self.kernel_size = patch_size
        self.stride = patch_size 
        self.region_size=region_size
        # 共享空间嵌入对象
        self.spatial_embedding = LUCCSpatialEmbedding(num_classes, embed_dim, patch_size,region_size)
        # 分别定义时间嵌入对象
        self.seq_temporal_embedding = LUCCTemporalEmbedding(embed_dim,seq_len) 
        self.pred_temporal_embedding = LUCCTemporalEmbedding(embed_dim,pred_len) 
        self.st_encoder = STEncoder(embed_dim, spatial_heads, temporal_heads, spatial_layers, temporal_layers, dropout, dim_feedforward)        
        self.st_decoder = STDecoder(embed_dim, spatial_heads, temporal_heads, spatial_layers, temporal_layers, dropout, dim_feedforward)

        self.deconv = nn.ConvTranspose3d(in_channels=num_classes, out_channels=num_classes, 
        kernel_size=(1, self.kernel_size[0], self.kernel_size[1]), 
                                 stride=(1, self.stride[0], self.stride[1]))          
        self.classifier = nn.Conv3d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1)

    def forward(self, x, y, return_logits=True, return_probs=False):
        B, T, H, W = y.shape
        # 使用共享的空间嵌入对象
        x_embedded = self.spatial_embedding(x)  # (B, seq_len, num_patches, embed_dim)          
        # 分别进行时间嵌入        
        x_embedded = self.seq_temporal_embedding(x_embedded)  # 对输入进行时间嵌入        
        memory = self.st_encoder(x_embedded)  # (B, seq_len, num_patches, embed_dim)  
        y = self.spatial_embedding(y)  # (B, pred_len, num_patches, embed_dim)      
        y = self.pred_temporal_embedding(y)   
        output = self.st_decoder(y, memory)  # (B, pred_len, num_patches, embed_dim)     
        # Reshape output to match target shape
        H_p, W_p = self.patch_size     
        num_patches_h = H // H_p
        num_patches_w = W // W_p      
        output = output.view(B, self.pred_len, num_patches_h, num_patches_w, self.embed_dim) 
        output = output.permute(0, 4, 1, 2, 3).contiguous()   
        logits = self.classifier(output)  # [B, num_classes, pred_len, num_patches_h, num_patches_w]       
        logits = self.deconv(logits)     
        if return_logits:            
            return logits
        elif return_probs:
            probs = F.softmax(logits, dim=1)  
            return probs
        else:
            preds = torch.argmax(logits, dim=1)  
            return preds


class LandTrans_ST_Decoder_Only(nn.Module):
    def __init__(self, seq_len, pred_len,region_size, patch_size, num_classes, embed_dim,
                spatial_heads, temporal_heads, spatial_layers, temporal_layers,
                dropout=0.1, dim_feedforward=2048):          
        super(LandTrans_ST_Decoder_Only, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.kernel_size = patch_size
        self.stride = patch_size

        self.spatial_embedding = LUCCSpatialEmbedding(num_classes, embed_dim, patch_size,region_size)
        # 分别定义时间嵌入对象
        self.seq_temporal_embedding = LUCCTemporalEmbedding(embed_dim,seq_len) 
        self.pred_temporal_embedding = LUCCTemporalEmbedding(embed_dim,pred_len) 

        self.st_decoder = STDecoder(embed_dim, spatial_heads, temporal_heads, spatial_layers, temporal_layers, dropout, dim_feedforward)     
        self.deconv = nn.ConvTranspose3d(in_channels=num_classes, out_channels=num_classes, 
        kernel_size=(1, self.kernel_size[0], self.kernel_size[1]), 
                                 stride=(1, self.stride[0], self.stride[1]))        
        self.classifier = nn.Conv3d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x,y, return_logits=True, return_probs=False):   
        B, T, H, W = x.shape    
        # 使用共享的空间嵌入对象
        x_embedded = self.spatial_embedding(x)  # (B, seq_len, num_patches, embed_dim) 
        x_embedded = self.seq_temporal_embedding(x_embedded)  # 对输入进行时间嵌入     
        y_embedded = self.spatial_embedding(y)  # (B, seq_len, num_patches, embed_dim)   
        y_embedded = self.pred_temporal_embedding(y_embedded)  # 对输入进行时间嵌入 
        # Generate causal mask
        mask=self.generate_square_subsequent_mask(y_embedded.size(1) * y_embedded.size(2)).to(y_embedded.device)       
        output = self.st_decoder(y_embedded, x_embedded, tgt_mask=mask)  # (B, pred_len, num_patches, embed_dim)        
        # Reshape output to match target shape
        H_p, W_p = self.patch_size     
        num_patches_h = H // H_p
        num_patches_w = W // W_p  
        output = output.view(B, self.pred_len, num_patches_h, num_patches_w, self.embed_dim) 
        output = output.permute(0, 4, 1, 2, 3).contiguous()   
        logits = self.classifier(output)  # [B, num_classes, pred_len, num_patches_h, num_patches_w]       
        logits = self.deconv(logits)        
        if return_logits:            
            return logits
        elif return_probs:
            probs = F.softmax(logits, dim=1)  
            return probs
        else:
            preds = torch.argmax(logits, dim=1)  
            return preds
