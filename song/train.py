import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from models.model import HandPoseModel

os.makedirs('models', exist_ok=True)

try:
    data = np.loadtxt('data/raw_data.csv', delimiter=',')
except Exception as e:
    print("오류!", e)
    exit()

# 마지막 열은 정답(Y), 나머지는 입력 데이터(X)
X = torch.tensor(data[:, :-1], dtype=torch.float32)
Y = torch.tensor(data[:, -1], dtype=torch.long)

model = HandPoseModel()
loss_fn = nn.CrossEntropyLoss() # 분류 문제용 오차 계산
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    pred = model(X)
    loss = loss_fn(pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'models/saved_weights.pth')
