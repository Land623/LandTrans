import torch
import torch.nn as nn
import torch.nn.functional as F

class LUCCTubeletEmbedding(nn.Module):
    def __init__(self, embed_dim, tubelet_size, input_size):
        super(LUCCTubeletEmbedding, self).__init__()
        self.embed_dim = embed_dim    
        self.tubelet_size = tubelet_size 
        self.input_size = input_size
        self.projection = nn.Conv3d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size,
        )     
        # Position embedding  
        num_patches = (input_size[0] // tubelet_size[0]) * (input_size[1] // tubelet_size[1]) * (input_size[2] // tubelet_size[2])
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, land_use_maps): 
        x = land_use_maps.permute(0, 4, 1, 2, 3) 
        x = self.projection(x) 
        x = x.flatten(2).transpose(1, 2)       
        # Add position embeddings  
        x = x + self.position_embeddings    
        return x

class LandTrans_Tubelet(nn.Module):
    def __init__(self,seq_len, pred_len,input_size,target_size,
                 tubelet_size,num_classes, embed_dim,  num_heads, num_encoder_layers=6,
                 num_decoder_layers=6,dropout=0.1,dim_feedforward=2048):
        super(LandTrans_Tubelet, self).__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.tubelet_size=tubelet_size
        self.input_tubelet_embedding = LUCCTubeletEmbedding(embed_dim, tubelet_size,input_size)  
        self.target_tubelet_embedding = LUCCTubeletEmbedding(embed_dim, tubelet_size,target_size)  
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)   
        self.classifierConv = nn.Conv3d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1)
       
        self.num_classes = num_classes
        self.seq_len=seq_len
        self.pred_len=pred_len
        self.embed_dim=embed_dim
        self.deconv = nn.ConvTranspose3d(
            in_channels=self.num_classes,
            out_channels=self.num_classes,
            kernel_size=tubelet_size,
            stride=tubelet_size,
        )  

    def forward(self, x, y, return_logits=True, return_probs=False):     
        B, T, H, W = x.shape    
        K = y.shape[1]
        # Embedding and Tubelet Embedding for encoder input
        x = self.embedding(x)  # (B, T, H, W, embed_dim)
        x = self.input_tubelet_embedding(x)  # (B, num_patches_encoder, embed_dim)        
        # Transformer Encoder  
        memory = self.encoder(x)  # (B, num_patches_encoder, embed_dim)        
        # Embedding and Tubelet Embedding for decoder input
        y = self.embedding(y)  # (B, K, H, W, embed_dim)
        y = self.target_tubelet_embedding(y)  # (B, num_patches_decoder, embed_dim)        
        # Transformer Decoder    
        output = self.decoder(y, memory)  # (B, num_patches_decoder, embed_dim)    
        T_p, H_p, W_p = self.tubelet_size
        num_patches_t = K // T_p
        num_patches_h = H // H_p
        num_patches_w = W // W_p
        # Classification      
        output = output.view(B, num_patches_t, num_patches_h, num_patches_w, self.embed_dim) 
        output = output.permute(0, 4, 1, 2, 3).contiguous()            
        logits = self.classifierConv(output)        
        logits = self.deconv(logits)   
        if return_logits:
            return logits
        elif return_probs:
            probs = F.softmax(logits, dim=1)
            return probs
        else:
            preds = torch.argmax(logits, dim=1)
            return preds


class LandTrans_Tubelet_Decoder_Only(nn.Module):
    def __init__(self,seq_len, pred_len,input_size,target_size,
                 tubelet_size,num_classes, embed_dim,  num_heads, 
                 num_layers=6,dropout=0.1,dim_feedforward=2048):
        super(LandTrans_Tubelet_Decoder_Only, self).__init__()
        
        self.embedding = nn.Embedding(num_classes, embed_dim)       
        self.input_tubelet_embedding = LUCCTubeletEmbedding(embed_dim, tubelet_size,input_size)  
        self.target_tubelet_embedding = LUCCTubeletEmbedding(embed_dim, tubelet_size,target_size)    

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.tubelet_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)        
        self.classifierConv = nn.Conv3d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1)        
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.pred_len=pred_len
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size         
        self.deconv = nn.ConvTranspose3d(
            in_channels=self.num_classes,
            out_channels=self.num_classes,
            kernel_size=tubelet_size,
            stride=tubelet_size,
        )
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x,y, return_logits=True, return_probs=False):  
        B, T, H, W = x.shape        
        # Embedding and Tubelet Embedding
        x = self.embedding(x)  # (B, T, H, W, embed_dim)
        x = self.input_tubelet_embedding(x)  # (B, num_patches, embed_dim)    
        y = self.embedding(y)  # (B, T, H, W, embed_dim)
        y = self.target_tubelet_embedding(y)  # (B, num_patches, embed_dim)     
        # Generate causal mask
        mask = self.generate_square_subsequent_mask(y.size(1)).to(y.device)      
        
        # Transformer Decoder (now acting as a self-attention mechanism) 
        output = self.tubelet_decoder(x, y, tgt_mask=mask)  # (B, num_patches, embed_dim)     
        # Reshape for classification
        T_p, H_p, W_p = self.tubelet_size
        num_patches_t = T // T_p
        num_patches_h = H // H_p
        num_patches_w = W // W_p     
        output = output.view(B, num_patches_t, num_patches_h, num_patches_w, self.embed_dim)    
        output = output.permute(0, 4, 1, 2, 3).contiguous()          
        # Classification
        logits = self.classifierConv(output)      
        logits = self.deconv(logits)  
        if return_logits:
            return logits
        elif return_probs:
            probs = F.softmax(logits, dim=1)
            return probs
        else:
            preds = torch.argmax(logits, dim=1)
            return preds
