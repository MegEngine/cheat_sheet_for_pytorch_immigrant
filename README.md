# Cheat Sheet for PyTorch Immigrant
# 一份给从 PyTorch 迁移过来的用户的小抄

## 如何从 PyTorch 迁移 weight 到 MegEngine 中？

首先确保两边的 Module 结构是类似的，在 PyTorch 里保存 state_dict，然后在 MegEngine 中将这个 weight 载入即可。

```python
# 在 PyTorch 中保存权重
import pickle
with open('torch-weight.pkl', 'wb') as f:
    states = net.state_dict()
    weights = {k: v.numpy() for k, v in states.items()}
    pickle.dump(weights, f)
```

```python
# 在 MegEngine 中读取权重
import pickle
with open('torch-weight.pkl', 'rb') as f:
    w = pickle.load(f)
weights = {}
for k, v in w.items():
    if k.endswith('bias') and v.ndim == 1:
        v = v.reshape(1, -1, 1, 1)
    weights[k] = v

net.load_state_dict(weights, strict=False)
```

在这个过程中可能会遇到一些 warning，可能是一些统计量没 load 成功之类的，一般问题不大。

原始帖子：https://discuss.megengine.org.cn/t/topic/1243
