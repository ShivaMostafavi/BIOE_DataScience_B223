import torch
from skorch import NeuralNetClassifier
from torch import nn, optim
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit


class CustomNN(nn.Module):
    def __init__(self, c_in=30, h_layer_sizes=None, nonlin=nn.LeakyReLU()):
        super(CustomNN, self).__init__()
        self.h_layer_sizes = h_layer_sizes
        self.c_in = c_in
        self.nonlin = nonlin
        self.h_layers = None
        h_layers_list = []

        if self.c_in is not None:
            self.input = nn.Linear(self.c_in, self.h_layer_sizes[0])
        if len(self.h_layer_sizes) > 1:
            prev = self.h_layer_sizes[0]
            for h_layer_size in h_layer_sizes:
                h_layers_list.append(nn.Linear(prev, h_layer_size))
                h_layers_list.append(nonlin)
                prev = h_layer_size
            self.h_layers = nn.Sequential(*h_layers_list)
        self.output = nn.Linear(self.h_layer_sizes[-1], 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, sample_weight=None, **kwargs):
        if self.c_in is None:
            self.c_in = x.size(dim=1)
            self.input = nn.Linear(self.c_in, self.h_layer_sizes[0])
        x = x.float()
        x = self.nonlin(self.input(x))
        if self.h_layers is not None:
            x = self.h_layers(x)
        x = self.softmax(self.output(x))
        return x


class CustomNNClassifier(NeuralNetClassifier):
    """
    Wrapper for the skorch NeuralNetClassifier to allow sample_weight.
    """
    def __init__(self, *args, criterion__reduce=False, **kwargs):
        super().__init__(*args, criterion__reduce=criterion__reduce, **kwargs)

    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        # override get_loss to use the sample_weight from X
        loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
        sample_weight = X['sample_weight']
        loss_reduced = (sample_weight * loss_unreduced).mean()
        return loss_reduced

    def fit(self, x, y, sample_weight=None):
        x = torch.tensor(x, device=self.device)
        if sample_weight is None:
            super().fit(x, y)
        else:
            sample_weight = torch.tensor(sample_weight, device=self.device)
            X_dict = {"x": x}
            # add sample_weight to the x dict
            X_dict['sample_weight'] = sample_weight
            super().fit(X_dict, y)


def get_nn(data_shape, cp, device):
    net = CustomNNClassifier(module=CustomNN, module__c_in=data_shape, module__h_layer_sizes=[50], max_epochs=100,
                             verbose=0, iterator_train__shuffle=True, criterion=nn.CrossEntropyLoss,
                             optimizer=optim.AdamW, train_split=ValidSplit(cv=0.1, stratified=True, random_state=42),
                             batch_size=16, lr=0.001, callbacks=[cp, EarlyStopping(patience=10)], device=device)
    return net
