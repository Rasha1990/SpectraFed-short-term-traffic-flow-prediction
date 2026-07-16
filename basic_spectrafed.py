import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
import copy
import numpy as np

# ==========================================
# 1. Core Spatiotemporal Model Architecture
# ==========================================
class GCNLayer(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(GCNLayer, self).__init__()
        self.gcn = GCNConv(in_features, hidden_dim)

    def forward(self, x, edge_index):
        return torch.relu(self.gcn(x, edge_index))

class GCNBiLSTMModel(nn.Module):
    """
    Basic GCN-BiLSTM model for spatiotemporal forecasting.
    This is the stripped-down core architecture provided for reproducibility.
    """
    def __init__(self, num_features, num_nodes, gcn_hidden=16, lstm_hidden=32, dropout=0.2):
        super(GCNBiLSTMModel, self).__init__()
        self.num_nodes = num_nodes
        self.gcn = GCNLayer(num_features, gcn_hidden)
        
        # BiLSTM input dimension is the flattened spatial features from GCN
        lstm_input_dim = gcn_hidden * num_nodes
        self.bilstm = nn.LSTM(lstm_input_dim, lstm_hidden, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # Output layer predicts 1 value per node for the next timestep
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_nodes)
        )

    def forward(self, x, edge_index):
        # x shape: [Batch, Time, Nodes, Features]
        B, T, N, F = x.shape
        
        # 1. Spatial Modeling: Apply GCN to each timestep independently
        x_reshaped = x.view(B * T, N, F)
        gcn_out_list = []
        for i in range(B * T):
            gcn_out_single = self.gcn(x_reshaped[i], edge_index) # [N, gcn_hidden]
            gcn_out_list.append(gcn_out_single)
            
        # Reshape back to temporal sequence: [B, T, N * gcn_hidden]
        gcn_out = torch.stack(gcn_out_list, dim=0).view(B, T, -1)
        
        # 2. Temporal Modeling: Process sequence with BiLSTM
        bilstm_out, _ = self.bilstm(gcn_out) # [B, T, lstm_hidden * 2]
        bilstm_out = self.dropout(bilstm_out)
        
        # 3. Prediction: Use the last timestep's hidden state to predict next step
        out = self.fc(bilstm_out[:, -1, :]) # [B, num_nodes]
        return out

# ==========================================
# 2. Minimal Federated Learning Components
# ==========================================
class FederatedClient:
    def __init__(self, model, data_loader, edge_index, client_id, device):
        self.model = copy.deepcopy(model).to(device)
        self.data_loader = data_loader
        self.edge_index = edge_index.to(device)
        self.client_id = client_id
        self.device = device

    def train(self, epochs=2, lr=1e-3):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            for batch in self.data_loader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                
                optimizer.zero_grad()
                out = self.model(x, self.edge_index)
                loss = criterion(out, y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
        return self.model.state_dict()

class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model

    def aggregate(self, client_states, weights):
        avg_state = {}
        for key in client_states[0].keys():
            # Skip batch norm running stats for simplicity in this basic version
            if 'running_mean' in key or 'running_var' in key:
                avg_state[key] = client_states[0][key].clone()
                continue
            
            avg_param = torch.zeros_like(client_states[0][key], dtype=torch.float32)
            for i, state in enumerate(client_states):
                avg_param += state[key].float() * weights[i]
            avg_state[key] = avg_param.to(client_states[0][key].dtype)
            
        self.global_model.load_state_dict(avg_state)
        return self.global_model

# ==========================================
# 3. Mock Dataset for Instant Reproducibility
# ==========================================
class MockTrafficDataset(Dataset):
    """
    A minimal mock dataset to prove the architecture works 
    without requiring external datasets (like PeMS).
    """
    def __init__(self, num_samples=100, num_nodes=10, his_length=12, num_features=1):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.his_length = his_length
        self.num_features = num_features
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Mock historical data: [Time, Nodes, Features]
        x = torch.randn(self.his_length, self.num_nodes, self.num_features)
        # Mock target: [Nodes] (predicting 1 feature for all nodes)
        y = torch.randn(self.num_nodes)
        return {'x': x, 'y': y}

# ==========================================
# 4. Main Execution (Basic Training Loop)
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Setup Mock Data
    num_nodes = 10
    num_features = 1
    train_dataset = MockTrafficDataset(num_samples=200, num_nodes=num_nodes, num_features=num_features)
    
    # Mock adjacency matrix (fully connected graph for demonstration)
    adj_matrix = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    edge_index = torch.tensor(np.vstack(np.where(adj_matrix > 0)), dtype=torch.long).to(device)

    # 2. Initialize Global Model
    global_model = GCNBiLSTMModel(
        num_features=num_features,
        num_nodes=num_nodes,
        gcn_hidden=16,
        lstm_hidden=32
    ).to(device)

    # 3. Federated Learning Setup
    num_clients = 2
    clients = []
    samples_per_client = len(train_dataset) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(train_dataset)
        
        # Simple non-overlapping data split
        client_subset = torch.utils.data.Subset(train_dataset, range(start_idx, end_idx))
        client_loader = DataLoader(client_subset, batch_size=16, shuffle=True)
        
        client = FederatedClient(global_model, client_loader, edge_index, client_id=i+1, device=device)
        clients.append(client)

    server = FederatedServer(global_model)

    # 4. Basic Federated Training Loop
    num_rounds = 3
    print("Starting basic federated training...")
    for round_num in range(num_rounds):
        client_states = []
        client_weights = []
        
        for client in clients:
            state = client.train(epochs=2, lr=1e-3)
            client_states.append(state)
            client_weights.append(len(client.data_loader.dataset))
            
        # Normalize weights and aggregate (FedAvg)
        weights = np.array(client_weights) / sum(client_weights)
        server.aggregate(client_states, weights)
        print(f"✅ Round {round_num+1}/{num_rounds} completed.")

    print(" Basic core model training finished. Ready for publication.")