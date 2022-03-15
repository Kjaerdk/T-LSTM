import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tlstm_model import TLSTMModel


class LitTLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.tlstm_model = TLSTMModel(input_dim, hidden_dim, output_dim, num_layers)
        self.save_hyperparameters()  # also saved the values in unpacked dict train_params

    def forward(self, x, time_deltas):
        yhat, hidden_sequences, (h_T, c_T) = self.tlstm_model(x, time_deltas)
        return yhat, hidden_sequences, (h_T, c_T)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop
        x, y, deltas = batch
        yhat, _, _ = self.tlstm_model(x, deltas)
        loss = F.mse_loss(yhat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("hp_metric", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.hparams.l_rate)
        return optimizer
