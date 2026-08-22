import torch
import torch.nn as nn

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
print(s)
s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
print(s)

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        print(x.dtype)

        x = self.relu(x)
        print(x.dtype)
        x = self.ln(x)
        print(x.dtype)

        x = self.fc2(x)
        print(x.dtype)

        return x

x = torch.rand(
2,2, device="mps"
    )

targets = torch.randint(
        0,
        64,
        ( 2,),
device="mps"
    )
model = ToyModel(2,64).to(device="mps")
dtype = torch.float16
with torch.autocast(device_type="mps", dtype=dtype):
        y = model(x)
        print(y.dtype)

        loss = torch.nn.functional.cross_entropy(y,targets)
        print(loss.dtype)

loss.backward()
for p in model.parameters():
    pass
    print(p.grad.dtype)