import torch
import torch.nn as nn
from safetensors.torch import load_file
import os
import json

class LSTMTimeSeries(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super(LSTMTimeSeries, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMTimeSeries(input_size=4, hidden_size=64, num_layers=1, output_size=1)
    model_path = os.path.join(model_dir, "model.safetensors")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        input_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Expecting [seq_len, 4]
        return input_data
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    with torch.no_grad():
        output = model(input_data)
    return output.cpu().numpy()

def output_fn(prediction, content_type):
    if content_type == "application/json":
        return json.dumps(prediction.tolist())
    raise ValueError(f"Unsupported content type: {content_type}")