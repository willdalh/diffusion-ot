import torch

def timestep_encoding(t, channels):
    """
    Creates a positional encoding identical to the one proposed in "Attention is all you need".
    
    Args:
        t (torch.Tensor): Tensor of shape (batch_size, 1)
        channels (int): Number of channels in the positional encoding
    """
    inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels))
    t_enc_a = torch.sin(t.repeat(1, channels//2) * inv_freq)
    t_enc_b = torch.cos(t.repeat(1, channels//2) * inv_freq)
    t_enc = torch.cat([t_enc_a, t_enc_b], dim=-1)
    return t_enc