from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.nn import functional as F
from pandas import DataFrame, Series
from torch import Tensor, nn
import torch
from typing import List
from tokenizer import (
    x_numerics,
    x_onehots,
    build_vocab,
    text_to_one_hot,
    numeric_to_log1p,
)
from torch.optim import Adam

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)
print(DEVICE)

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual) -> Tensor:
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

class _Dense(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_size, 1024)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(1024, 1024)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(input_size + 1024, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act1(self.lin1(x))
        h = self.act2(self.lin2(h))
        h = torch.cat([h, x], dim=1)
        out = self.output(h)
        return out[:, 0]

class Model():
    def __init__(self) -> None:
        pass
    
    def build(self, x: DataFrame):
        onehots = x[x_onehots]
        numerics = x[x_numerics]
        self.vocab = build_vocab(onehots)
        t_onehots = text_to_one_hot(onehots, self.vocab)
        t_numerics = numeric_to_log1p(numerics)
        t_input = torch.cat([t_onehots, t_numerics], dim=1)
        self.input_size = t_input.shape[1]
        self.model = _Dense(self.input_size)

    def train(self, x: DataFrame, y: Series, epochs: int=100, batch_size: int=64):
        self.model.train()
        t_input = self.process(x)
        t_y = numeric_to_log1p(y).to(DEVICE)
        
        criterion = RMSLELoss()
        optimizer = Adam(self.model.parameters(), lr=0.0001)

        best = None
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(t_input), batch_size):
                optimizer.zero_grad()
                t_y_pred = self.model(t_input[i:i+batch_size])
                loss = criterion(t_y_pred, t_y[i:i+batch_size])
                total_loss += torch.sum(loss)
                loss.backward()
                optimizer.step()
            print(f"epoch: {epoch}, loss: {total_loss}")
            if not best or total_loss < best:
                best = total_loss
                torch.save(self, f"saved_models/dense_{best}.pickle")
        return best

    def process(self, x: DataFrame) -> Tensor:
        onehots = x[x_onehots]
        numerics = x[x_numerics]
        t_onehots = text_to_one_hot(onehots, self.vocab)
        t_numerics = numeric_to_log1p(numerics)
        t_input = torch.cat([t_onehots, t_numerics], dim=1)
        return t_input.to(DEVICE)

    def predict(self, t: Tensor) -> Tensor:
        self.model.eval()
        out: Tensor = self.model(t)
        return out.expm1().to(DEVICE)
    
    def __call__(self, x: DataFrame) -> List[float]:
        t = self.process(x)
        t = self.predict(t)
        return t.tolist()

