
from argparse import ArgumentParser
import datetime as dt
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data_utils import data_dict_loader, BucketBatchSampler, BucketDataset
from lightning import LitTLSTM


def main(args):

    dict_args = vars(args)
    print(dict_args)

    # Check n_layers corresponds for len(hidden_dim)
    if not len(dict_args['hidden_dim']) == dict_args['num_layers']:
        print('Layers and length of list with hidden dimensions must match')
        return None

    # Step 1: Load data - DONE
    data_dict = data_dict_loader(dict_args['prep_data'])
    # Step 2: Make custom batch sampler (to avoid padding)
    bucket_batch_sampler = BucketBatchSampler(data_dict['X'], dict_args['batch_size'])
    # Step 3: Put data into a Dataset object - DONE
    bucket_dataset = BucketDataset(data_dict['X'], data_dict['Y'], data_dict['time_delta'])
    # Step 4: Make DataLoader
    dataloader = DataLoader(bucket_dataset, batch_size=1, batch_sampler=bucket_batch_sampler, shuffle=False,
                            num_workers=0, drop_last=False)

    dict_args['input_dim'] = dataloader.dataset.inputs[0].shape[1]
    dict_args['output_dim'] = dataloader.dataset.targets[0].shape[1]

    # Variables for easy save results
    now = dt.datetime.now()
    date = str(now)[8:10] + str(now.strftime("%B")) + str(now)[11:13] + '.' + str(now)[14:16]
    path = 'Hidden' + str(dict_args['hidden_dim']) + 'Layers' + str(dict_args['num_layers']) + '_' + date

    # init model
    tlstm = LitTLSTM(**dict_args)
    if dict_args['train_model']:
        trainer = pl.Trainer(default_root_dir='/path/to/logs/' + path, max_epochs=dict_args['epochs'])
        trainer.fit(tlstm, dataloader)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--prep_data', type=bool, default=False)
    parser.add_argument('--train_model', type=bool, default=True)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--l_rate', type=float, default=0.01)

    parser.add_argument('--hidden_dim', type=list, default=[10, 5])
    parser.add_argument('--num_layers', type=int, default=2)

    args = parser.parse_args()

    main(args)


"""
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
