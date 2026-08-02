import os
import re
from typing import Optional
from torch.utils.data import Dataset
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class ArcMarginProduct(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        s=30.0,
        m=0.50
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x, label=None):
        cosine = F.linear(
            F.normalize(x),
            F.normalize(self.weight)
        )
        if label is None:
            return cosine * self.s
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(
            1,
            label.view(-1,1),
            1
        )
        theta = torch.acos(
            torch.clamp(cosine,-1+1e-7,1-1e-7)
        )
        target = torch.cos(theta+self.m)
        logits = cosine*(1-one_hot)+target*one_hot
        return logits*self.s

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))

class ProjectionHead(nn.Module):
    """投影头。"""
    def __init__(
            self,
            input_dim=384,
            proj_dim=256,
            dropout=0.2
        ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, proj_dim)
        )

    def forward(self,x):
        x=self.net(x)
        return nn.functional.normalize(x,p=2,dim=-1)

class PrototypeClassifier(nn.Module):
    """轻量多语言原型分类头。"""
    def __init__(
        self,
        feat_dim,
        num_classes,
        temperature=16.0
    ):
        super().__init__()
        self.prototype = nn.Parameter(
            torch.randn(
                num_classes,
                feat_dim
            )
        )
        nn.init.xavier_uniform_(self.prototype)
        self.temperature=temperature

    def forward(self,x):
        x=F.normalize(x,p=2,dim=1)
        p=F.normalize(
            self.prototype,
            p=2,
            dim=1
        )
        logits=torch.matmul(
            x,
            p.t()
        )
        return logits*self.temperature

class EmotionClassifier(nn.Module):
    """轻量多语言情感分类头。"""
    def __init__(
        self,
        input_dim=384,
        num_classes=19,
        hidden_dim=384,
        proj_dim=256,
        dropout=0.3
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.Sequential(
            ResidualBlock(hidden_dim,dropout),
            ResidualBlock(hidden_dim,dropout)
        )
        self.projector = ProjectionHead(
            hidden_dim,
            proj_dim,
            dropout
        )
        self.classifier = PrototypeClassifier(
            proj_dim,
            num_classes
        )
        self.arcface = ArcMarginProduct(
            proj_dim,
            num_classes
        )

    def forward(self,x,label=None):
        x = self.input_proj(x)
        x = self.encoder(x)
        feat = self.projector(x)
        logits = self.classifier(feat)
        if label is not None:
            arc_logits = self.arcface(feat, label)  # 带标签时返回ArcFace logits
        else:
            arc_logits = self.arcface(feat)         # 无标签时返回常规cosine
        return logits, feat, arc_logits
