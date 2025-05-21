import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class EWMA_Smooth(nn.Module):
    def __init__(self, N, M, dim_feedforward=128, init_beta=0.9, nhead=4, d_model=64, dropout=0.1):
        super(EWMA_Smooth, self).__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))
        self.register_buffer('j_values', torch.arange(M - 1, -1, -1).float())

        self.N = N
        self.M = M
        self.d_model = d_model
        

        self.embedding = nn.Linear(1, d_model)
        
  
        self.time_pos = nn.Parameter(torch.randn(1, M, d_model))
        self.space_pos = nn.Parameter(torch.randn(N, 1, d_model))

        self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.space_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.output = nn.Linear(d_model, 1)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):

        B, N, M = x.size()
        
        alphas = torch.sigmoid(self.beta * self.j_values[:M])
        weights = (1 - alphas).cumprod(dim=0).roll(1, 0)
        weights[0] = 1.0

        weighted_sum = torch.einsum('bnm,m->bnm', x, weights)
        cum_weights = weights.cumsum(0)
        trend = weighted_sum / cum_weights.clamp(min=1e-7)
        

        x_emb = self.embedding(trend.unsqueeze(-1))  
        

        x_emb = x_emb + self.time_pos.unsqueeze(0) + self.space_pos.unsqueeze(0)  
        

        time_input = x_emb.reshape(B * N, M, self.d_model)
        time_input = time_input.permute(1, 0, 2)
        time_attn_out, _ = self.time_attn(time_input, time_input, time_input)
        time_out = time_input + time_attn_out  
        time_out = time_out.permute(1, 0, 2).reshape(B, N, M, self.d_model)
        
      
        space_input = x_emb.permute(0, 2, 1, 3).reshape(B * M, N, self.d_model)
        space_input = space_input.permute(1, 0, 2)  
        space_attn_out, _ = self.space_attn(space_input, space_input, space_input)
        space_out = space_input + space_attn_out 
        space_out = space_out.permute(1, 0, 2).reshape(B, M, N, self.d_model).permute(0, 2, 1, 3)
        

        combined = torch.cat([time_out, space_out], dim=-1)
        gate_weight = self.gate(combined)  
        fused = gate_weight * time_out + (1 - gate_weight) * space_out
        
 
        fused = fused + self.ffn(fused)
        fused = self.norm(fused)
        output = self.output(fused).squeeze(-1) 
        
        return output + x

    
class DNN(nn.Module):
    def __init__(self, num_points, hidden_size, output_points, drop_out, num_heads=1):
        super(DNN, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

   
        self.fc1 = nn.Linear(num_points, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)



        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)

        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.ln4 = nn.LayerNorm(hidden_size // 2)

        self.fc5 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.ln5 = nn.LayerNorm(hidden_size // 2)


        self.fc6 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.ln6 = nn.LayerNorm(hidden_size // 2)

        self.fc_output = nn.Linear(hidden_size // 2, output_points)

        self.saved_attention_weights = None


    def forward(self, x):


        x = x.permute(0, 2, 1)  

        x1 = self.fc1(x)         
        x1 = self.ln1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

    
        x2 = self.fc2(x1)         
        x2 = self.ln2(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)

    
        x3 = self.fc3(x2)         
        x3 = self.ln3(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)

 
        x4 = self.fc4(x3)        
        x4 = self.relu(x4)
        x4 = self.dropout(x4)
        x4 = x4 + x3               

        x5 = self.fc5(x4)          
        x5 = self.ln5(x5)
        x5 = self.relu(x5)
        x5 = self.dropout(x5)

        x6 = self.fc6(x5)          
        x6 = self.ln6(x6)
        x6 = self.relu(x6)
        x6 = self.dropout(x6)
        x6 = x6 + x5                
        output = self.fc_output(x6)  

        output = output.permute(0, 2, 1)

        return output

class AFD_STA_Model(nn.Module):

    def __init__(self, time_steps, num_points, hidden_size, output_points, exp_k, drop_out):
        super(AFD_STA_Model, self).__init__()
        self.ewma = EWMA_Smooth(num_points,time_steps)
        self.dnn = DNN(num_points, hidden_size, output_points, drop_out)

    def forward(self, x):

        x_linear = self.ewma(x)  
        x_dnn = self.dnn(x_linear) 
        output = x_dnn
        return output

def compute_rmse(outputs, targets):
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs, dtype=torch.float32).to(device)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32).to(device)

    mse = torch.mean((outputs - targets) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()

def train_model(params, train_data, train_labels):

    model =AFD_STA_Model(
        time_steps=params['M'],
        num_points=train_data.shape[1],
        hidden_size=params['hidden_layers'],
        output_points=params['L']+1,
        exp_k=params['exp_k'],
        drop_out=params['drop_out']
    )

    model.to(device)

    model_name = model.__class__.__name__
    model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"{model_name}.pth")

    mse_criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    avg_loss = 0.0
    avg_rmse = 0.0

    lambda_diag = params['lambda_diag']  

    for epoch in range(params['epochs']):
        model.train()
        total_loss = 0.0
        total_rmse = 0.0

        for batch_data, batch_labels in train_loader:

            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_data)
            
            mse_loss = mse_criterion(outputs, batch_labels)
            
            
            loss = mse_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            batch_size_current = batch_data.size(0)
            total_loss += loss.item() * batch_size_current
            rmse = compute_rmse(outputs.detach(), batch_labels.detach())
            total_rmse += rmse * batch_size_current



        avg_loss = total_loss / len(train_loader.dataset)
        avg_rmse = total_rmse / len(train_loader.dataset)

    

    torch.save(model, model_save_path)
    print(f"Model is saved at: {model_save_path}")

    return avg_loss, model_save_path