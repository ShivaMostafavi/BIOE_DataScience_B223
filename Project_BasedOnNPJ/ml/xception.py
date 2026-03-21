from skorch import NeuralNetClassifier
from torch import nn, optim
from skorch.callbacks import EarlyStopping, EpochScoring, Checkpoint, LRScheduler
from skorch.dataset import ValidSplit
from tsai.models.XceptionTimePlus import XceptionTimePlus
from ml.nn import CustomNNClassifier


class CustomXceptionTime(XceptionTimePlus):
    """
    Custom xception to allow sample_weight.
    """
    def __init__(self, *args, **kwargs):
        super(CustomXceptionTime, self).__init__(*args, **kwargs)

    def forward(self, x, sample_weight=None, **kwargs):
        return super().forward(x)


def get_xception(data_shape, cp, device): ## Max epochs 200
    net = CustomNNClassifier(module=CustomXceptionTime, module__c_in=data_shape, module__c_out=2, module__nf=8,
                             module__act=nn.LeakyReLU, max_epochs=200, iterator_train__shuffle=True,
                             criterion=nn.CrossEntropyLoss, optimizer=optim.AdamW,
                             train_split=ValidSplit(cv=0.1, stratified=True, random_state=42), batch_size=16, lr=0.001,
                             callbacks=[cp,
                                        EarlyStopping(patience=100),
                                        LRScheduler(policy=optim.lr_scheduler.OneCycleLR,
                                                    monitor="valid_loss", max_lr=0.001,
                                                    epochs=1,
                                                    steps_per_epoch=200,
                                                    div_factor=20,
                                                    final_div_factor=2000,
                                                    step_every="epoch")],
                             device=device)
    return net
