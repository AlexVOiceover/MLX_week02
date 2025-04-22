import torch


class QryTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


class DocTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


qryTower = QryTower()
docTower = DocTower()

qry = torch.randn(1, 10)  # 1: query, 10-dim embedding
pos = torch.randn(1, 10)  # 1: positive doc, 10-dim embedding
neg = torch.randn(1, 10)  # 1: negative doc, 10-dim embedding

qry = qryTower(qry)
pos = docTower(pos)
neg = docTower(neg)

dst_pos = torch.nn.functional.cosine_similarity(qry, pos)
dst_neg = torch.nn.functional.cosine_similarity(qry, neg)
dst_dif = dst_pos - dst_neg
dst_mrg = torch.tensor(0.2)

loss = torch.max(torch.tensor(0.0), dst_mrg - dst_dif)
loss.backward()
