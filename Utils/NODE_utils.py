import torch
import torch.nn as nn

def init_weights(m, activation_type):
    if isinstance(m, nn.Linear):
        if activation_type.lower() in ['relu', 'leakyrelu']:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        else:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def normalize(out, eps=1e-8):
    """
    Project the output tensor on the simplex to ensure it sums to 1 along the last dimension.
    """
    out = torch.relu(out)
    sums = out.sum(dim=-1, keepdim=True)
    
    mask_zero = (sums < eps)
    out_normalized = out / (sums + eps)
    uniform = torch.ones_like(out_normalized) / out.shape[-1]
    out_final = torch.where(mask_zero, uniform, out_normalized)
    return out_final