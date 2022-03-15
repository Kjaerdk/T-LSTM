
import os
import datetime as dt
import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from data_utils import data_dict_loader, BucketBatchSampler, BucketDataset
from lightning import LitTLSTM

PREPDATA = False
epochs = 100
DIR = os.getcwd()


def objective(trial: optuna.trial.Trial) -> float:

    # Optimize over batchsize and epochs
    batch_size = trial.suggest_int('batch_size', low=50, high=50, step=10)
    #epochs = trial.suggest_int('epochs', low=100, high=100, step=1)

    # We optimize the number of layers, hidden units in each layer and dropouts.
    num_layers = trial.suggest_int("n_layers", 1, 2)
    l_rate = trial.suggest_uniform("lr", 1e-5, 1e-1)

    which_list = trial.suggest_int('which_list', 0, 2)
    hidden_dim_options = [[20, 10], [10, 5], [5, 3]] if num_layers == 2 else [[15], [10], [5]]
    hidden_dim = hidden_dim_options[which_list]

    setup_dict = dict(batch_size=batch_size, prep_data=PREPDATA, l_rate=l_rate, epochs=epochs,
                      num_layers=num_layers, hidden_dim=hidden_dim)

    # Step 1: Load data - DONE
    data_dict = data_dict_loader(setup_dict['prep_data'])
    # Step 2: Make custom batch sampler (to avoid padding)
    bucket_batch_sampler = BucketBatchSampler(data_dict['X'], setup_dict['batch_size'])
    # Step 3: Put data into a Dataset object - DONE
    bucket_dataset = BucketDataset(data_dict['X'], data_dict['Y'], data_dict['time_delta'])
    # Step 4: Make DataLoader
    dataloader = DataLoader(bucket_dataset, batch_size=1, batch_sampler=bucket_batch_sampler, shuffle=False,
                            num_workers=0, drop_last=False)

    setup_dict['input_dim'] = dataloader.dataset.inputs[0].shape[1]
    setup_dict['output_dim'] = dataloader.dataset.targets[0].shape[1]

    # Variables for easy save results
    now = dt.datetime.now()
    date = str(now)[8:10] + str(now.strftime("%B")) + str(now)[11:13] + '.' + str(now)[14:16]
    path = 'Hidden' + str(setup_dict['hidden_dim']) + 'Layers' + str(setup_dict['num_layers']) + '_' + date

    print(setup_dict)
    model = LitTLSTM(**setup_dict)

    trainer = pl.Trainer(
        default_root_dir='/path/to/logs/' + path,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="train_loss")],
        max_epochs=setup_dict['epochs'],
    )
    trainer.logger.log_hyperparams(setup_dict)
    trainer.fit(model, dataloader)

    return trainer.callback_metrics["train_loss"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )
    study_name = 'example-study'  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(direction="minimize", pruner=pruner, study_name=study_name,
                                storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=10, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


"""
# How to load existing study and check out results:
import optuna

loaded_study = optuna.load_study(study_name=study_name, storage=storage_name)
df = loaded_study.trials_dataframe()

# How to get h_params from a checkpoint
path = '/path/to/logs/path/lightning_logs/version_0/checkpoints'
model = LitTLSTM.load_from_checkpoint(checkpoint_path=path + '/epoch=0-step=88.ckpt')
>>> model.hparams
    "batch_size":  10
    "epochs":      1
    "hidden_dim":  3
    "input_dim":   319
    "l_rate":      0.1
    "momentum":    0.9
    "num_layers":  2
    "output_dim":  82
    "prep_data":   False
    "train_model": True

"""
