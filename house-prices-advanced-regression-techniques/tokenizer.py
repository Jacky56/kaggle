from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.nn import functional as F
from pandas import DataFrame, Series
from torch import Tensor
import torch
from typing import Dict

UNK = "<unk>"

x_onehots = [
    "MSSubClass",
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1", # ???
    "Condition2", # ???
    "BldgType",
    "HouseStyle",
    "OverallQual",
    "OverallCond",
    "YearBuilt", # year of build, maybe requied to null
    "YearRemodAdd", # year of remoddel, maybe requied to null
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageType",
    "GarageYrBlt", # year of build, maybe requied to null
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MoSold",
    "YrSold",
    "SaleType",
    "SaleCondition",
]   

x_numerics = [
    "LotFrontage",
    "LotArea",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
]

y_numerics = "SalePrice"

def build_vocab(corpus: DataFrame, default: str=UNK) -> Dict[str, Vocab]:
    d = {}
    for column in corpus:
        vocab = build_vocab_from_iterator(corpus[column].map(lambda s: str(s).split()), specials=[UNK])
        vocab.set_default_index(vocab[default])
        d[column] = vocab
    return d

def text_to_one_hot(categorical: DataFrame, vocab: Dict[str, Vocab]) -> Tensor:
    big_t = []
    for column in categorical:
        li = categorical[column].fillna(UNK)
        tokenizer = vocab[column]
        t = Tensor(li.map(lambda s: tokenizer(str(s).split())[0])).to(torch.int64) # each str shouuld only have 1 word hence [0]
        t = F.one_hot(t, num_classes=len(tokenizer))
        big_t.append(t)
    big_t = torch.cat(big_t, dim=1)
    return big_t

def numeric_to_log1p(numerical: DataFrame | Series) -> Tensor:
    numerical = numerical.fillna(0)
    t = torch.log1p(Tensor(numerical.to_numpy()))
    return t
