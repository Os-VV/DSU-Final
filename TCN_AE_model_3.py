import torch
from torch.nn import MultiheadAttention
from torch.utils.data import Dataset
import torch.nn as nn
from pytorch_tcn import TCN
import polars as pl
import joblib
from sklearn.preprocessing import StandardScaler


class CustomDataset(Dataset):

    def __init__(self, 
                 data_path, 
                 win_size, 
                 stride,
                 train = True
                 ):
        
        self.win_size = win_size
        self.stride = stride
        self.datapath = data_path

        df = pl.read_parquet(data_path, use_pyarrow=True, memory_map = True)
        df = df.drop(['Timestamp'])
        if train:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df)
            joblib.dump(scaler, 'tina_train_scaler_fit.joblib')
        else:
            loaded_scaler = joblib.load('tina_train_scaler_fit.joblib')
            scaled = loaded_scaler.transform(df)
        self.features = torch.from_numpy(scaled).float()
        data_length = self.features.shape[0]
        self.num_windows = (data_length - self.win_size) // self.stride + 1  
        
    def __len__(self):
        return self.num_windows
      
    def __getitem__(self, idx):

        start_idx = idx * self.stride
        end_idx = start_idx + self.win_size

        x = self.features[start_idx:end_idx]
        y = torch.clone(x)
       
        return x, y

class encoder_decoder_tcn(nn.Module):
    def __init__(self):
        super(encoder_decoder_tcn, self).__init__()
        
        self.local_encoder = TCN(
            num_inputs = 103,
            num_channels= [103, 128, 256, 512, 256, 128],
            kernel_size = 2,
            dilations = [1,2,4,6,8, 16],
            dropout = 0.25,
            causal = False,
            use_norm = 'weight_norm', 
            activation = 'leaky_relu',
            use_skip_connections = True,
            input_shape = 'NLC',
            kernel_initializer = 'xavier_uniform'
            )
        
        self.global_encoder = TCN(
            num_inputs = 103,
            num_channels= [103, 128, 256, 512, 256, 128],
            kernel_size = 8,
            dilations=[1, 8, 16, 32, 64, 128],
            dropout = 0.25,
            causal = False,
            use_norm = 'weight_norm',
            activation = 'leaky_relu',
            use_skip_connections = True,
            input_shape = 'NLC',
            kernel_initializer = 'xavier_uniform'
            )
        
        self.decoder = TCN(
            num_inputs = 256,
            num_channels= [512, 1024, 512, 256, 256, 103],
            kernel_size = 2,
            dilations = [1, 4, 8, 16, 32, 64],
            dropout = 0.25,
            causal = False,
            use_norm = 'weight_norm',
            activation = 'leaky_relu',
            use_skip_connections = True,
            input_shape = 'NLC',
            kernel_initializer = 'xavier_uniform'
            )
              
        self.emb_dim = 256
        self.l_norm = nn.LayerNorm(self.emb_dim)
        self.output_head = nn.Linear(103, 103)
                    
        self.mha = MultiheadAttention(embed_dim = self.emb_dim, num_heads = 4, batch_first = True, dropout= 0.1)

    def forward(self, input):
        local_enc_out = self.local_encoder(input)
        global_enc_out = self.global_encoder(input)
        enc_comb_out = torch.concat((local_enc_out, global_enc_out), dim = -1)
        enc_comb_out = self.l_norm(enc_comb_out)
        attn_out, _= self.mha(enc_comb_out, enc_comb_out, enc_comb_out, need_weights = False, attn_mask=None)
        attn_out = enc_comb_out + attn_out
        dec_out = self.decoder(attn_out)
        out = self.output_head(dec_out)

        return out   