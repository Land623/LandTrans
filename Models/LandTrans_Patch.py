import torch
import torch.nn as nn
import torch.nn.functional as F


class LUCCPatchEmbedding(nn.Module):
    def __init__(self,num_classes, embed_dim, patch_size, input_size):
        super(LUCCPatchEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_classes, embed_dim)   
        self.patch_size = patch_size 
        self.input_size = input_size
        self.STConv3= nn.Conv3d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            padding=1
        )
        self.STConv2= nn.Conv3d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=(2,3,3),
            padding=1
        )
        self.projection = nn.Conv3d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=(input_size[0],patch_size[0],patch_size[1]),
            stride=(input_size[0], patch_size[0],patch_size[1]),
        )
        # Position embedding  
        num_patches = (input_size[1] // patch_size[0]) * (input_size[2] // patch_size[1])
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, embed_dim))   
        

    def forward(self, land_use_maps):  
        land_use_maps = self.embedding(land_use_maps)            
        # Reshape to (B, embed_dim, T, H, W) to apply Conv3d       
        x = land_use_maps.permute(0, 4, 1, 2, 3)         
        # Apply Conv3d to get embeddings   
        if x.shape[2]==2:
            x = self.STConv2(x)
        else:
            x = self.STConv3(x)
        x = self.projection(x)  # (B, embed_dim, T//T_p, H//H_p, W//W_p)             
        # Reshape to (num_patches,B, embed_dim)
        x = x.flatten(2).transpose(1, 2)       
        # Add position embeddings 
        x = x + self.position_embeddings        
        return x

class LandTrans_Patch(nn.Module):
    def __init__(self, seq_len, pred_len, patch_size,input_size,target_size, num_classes, embed_dim, 
                 num_heads, num_encoder_layers,num_decoder_layers, 
                 dropout=0.1, dim_feedforward=2048):
        super(LandTrans_Patch, self).__init__()      
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_classes = num_classes 
        self.patch_size = patch_size
        self.kernel_size = (pred_len,patch_size[0],patch_size[1])
        self.stride=(pred_len,patch_size[0],patch_size[1])     
        self.input_seq_embeding=LUCCPatchEmbedding(num_classes, embed_dim, patch_size, input_size)
        self.target_seq_embeding=LUCCPatchEmbedding(num_classes, embed_dim, patch_size, target_size)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)      
        self.deconv = nn.ConvTranspose3d(in_channels=num_classes, out_channels=num_classes, kernel_size=self.kernel_size, stride=self.stride)
        self.classifier = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1)

    def forward(self, x,y, return_logits=True, return_probs=False):  
        B, T, H, W = y.shape      
        x=self.input_seq_embeding(x)        
        y=self.target_seq_embeding(y)        
        memory = self.encoder(x)  # [num_frames * num_patches, batch_size, embed_dim]  
        output = self.decoder(y, memory)  # [num_output_frames * num_patches, batch_size, embed_dim]           
        # Reshape x to match the target shape
        H_p, W_p = self.patch_size     
        num_patches_h = H // H_p
        num_patches_w = W // W_p   
     
        output = output.view(B, num_patches_h, num_patches_w,self.embed_dim) 
        output = output.permute(0, 3, 1, 2).contiguous() 
        logits = self.classifier(output) 
        logits = logits.unsqueeze(2)  
        logits = self.deconv(logits)  
        if return_logits:
            return logits
        elif return_probs:
            probs = F.softmax(logits, dim=1)  # 在类别维度上应用softmax
            return probs
        else:
            preds = torch.argmax(logits, dim=1)  # 在类别维度上应用argmax
            return preds   

class LandTrans_Patch_Decoder_Only(nn.Module):
    def __init__(self, seq_len, pred_len, patch_size,input_size,target_size,
                  num_classes, embed_dim,
                num_heads, num_layers, dropout=0.1, dim_feedforward=2048):    
        super(LandTrans_Patch_Decoder_Only, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.patch_size = patch_size      
        self.kernel_size = (pred_len,patch_size[0],patch_size[1])
        self.stride=(pred_len,patch_size[0],patch_size[1])    
        self.input_seq_embeding=LUCCPatchEmbedding(num_classes, embed_dim, patch_size, input_size)
        self.target_seq_embeding=LUCCPatchEmbedding(num_classes, embed_dim, patch_size, target_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.deconv = nn.ConvTranspose3d(in_channels=num_classes, out_channels=num_classes, kernel_size=self.kernel_size, stride=self.stride)
        self.classifier = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, y, return_logits=True, return_probs=False):    
        B, T, H, W = x.shape  
        x_embedded = self.input_seq_embeding(x) 
        y_embedded = self.target_seq_embeding(y)        
        # Generate causal mask
        mask = self.generate_square_subsequent_mask(y_embedded.size(1)).to(y_embedded.device)  
        # Transformer Decoder (now acting as a self-attention mechanism)     
        output = self.decoder(x_embedded, y_embedded, tgt_mask=mask)  # (B, num_patches, embed_dim)  
       
        H_p, W_p = self.patch_size
        num_patches_h = H // H_p
        num_patches_w = W // W_p  
        output = output.view(B,num_patches_h, num_patches_w, self.embed_dim) 
        output = output.permute(0, 3, 1, 2).contiguous()     
        logits = self.classifier(output) 
        logits = logits.unsqueeze(2) 
        logits = self.deconv(logits)   
        if return_logits:            
            return logits
        elif return_probs:
            probs = F.softmax(logits, dim=1)  
            return probs
        else:
            preds = torch.argmax(logits, dim=1)  
            return preds
