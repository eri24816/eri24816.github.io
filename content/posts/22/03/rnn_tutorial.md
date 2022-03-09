---
title: "PyTorch RNN Tutorial"
date: 2022-03-09T12:54:18+08:00
draft: false
image: https://i.imgur.com/X6kty5v.jpg
mathjax: true
---


This tutorial will demonstrate how to build and train a simple RNN model with PyTorch.

### Imports
Import the modules we will need.
```python
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
```

### Prepare the training data
First, we need the training data. 
Here we generate a sine wave of $\sin(t)$, from $t=0$ to $t=80$, and sample peirod $=0.2$. Then add a little noise to it.

```python
data = np.sin(np.arange(0,80,0.2))
data += (np.random.random(data.shape)-0.5)*0.2
plt.plot(data)
```
![Image](https://i.imgur.com/07PP9iu.jpg#centers)

Later, we will build the RNN model to learn on the training data.

```python
data = torch.tensor(data, dtype = torch.float).unsqueeze(-1).unsqueeze(-1)
x = data[:-1]
y = data[1:]
```
