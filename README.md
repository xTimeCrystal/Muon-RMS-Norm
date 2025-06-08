# Muon RMS-Norm
This version of Muon converges slightly faster than the Muon from modded-nanogpt when training RWKV 7 (roughly 10% faster). 

The change is RMS-Norm after orthogonalization over the **first** dimension of the weight matrix (**last dimension of nn.Linear**). The code here **assumes you store the weights like nn.Linear** i.e. used like `x = x @ W.T`. 

This is untested for other architectures and it is unknown why this works better. 

Example usage:
```python
muon_filter = lambda p: p.ndim == 2
optimizer = Muon([p for p in model.parameters() if muon_filter(p)], lr=1e-3, momentum=0.95, weight_decay=1e-2)
```
