import torch
import torch_xla
import torch_xla.core.xla_model as xm

print("Starting...")
print("Supported xla devices")
print(xm.get_xla_supported_devices())
t = torch.randn(2, 2, device=xm.xla_device())
print(t.device)
print(t)
