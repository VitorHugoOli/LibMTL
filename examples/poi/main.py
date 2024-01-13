import dataclasses

import torch.nn as nn

import LibMTL.architecture as architecture_method
import LibMTL.weighting as weighting_method
from LibMTL import Trainer
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric
from LibMTL.utils import set_random_seed, set_device
from examples.poi.create_dataset import creating_dataloaders
from examples.poi.models.pcg_nn import PCGNN
from examples.poi.models.poi_rgnn import POIRGNN


@dataclasses.dataclass
class TaskManager:
    name: str
    evaluation_metrics: dict
    model: type(nn.Module)
    in_channels: int
    out_channels: int
    num_classes: int
    dropout: float


CONFIGS_TASKS = {
    'poi': {
        'in_channels': 1,
        'out_channels': 1,
        'num_classes': 1,
        'dropout': 0.5
    },
    'pcg': {
        'in_channels': 1,
        'out_channels': 1,
        'num_classes': 1,
        'dropout': 0.5
    }
}


def parse_args(parser):
    parser.add_argument('--bs', default=64, type=int, help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
    parser.add_argument('--dataset_path', default='data/', type=str, help='Datasets path')
    return parser.parse_args()


def initialize_tasks():
    return [
        TaskManager(name='poi',
                    evaluation_metrics={
                        'metrics': ['Acc'],
                        'metrics_fn': AccMetric(),
                        'loss_fn': CELoss(),
                        'weight': [1]
                    },
                    model=POIRGNN,
                    **CONFIGS_TASKS['poi']),
        TaskManager(name='pcg',
                    evaluation_metrics={
                        'metrics': ['Acc'],
                        'metrics_fn': AccMetric(),
                        'loss_fn': CELoss(),
                        'weight': [1]
                    },
                    model=PCGNN,
                    **CONFIGS_TASKS['pcg'])
    ]


def prepare_dataloaders(tasks_managers):
    data_loader, _ = creating_dataloaders()
    loaders = {}
    for dataset in ['train', 'val', 'test']:
        loaders[dataset] = {task.name: data_loader[task.name][dataset] for task in tasks_managers}
    return loaders['train'], loaders['val'], loaders['test']


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, inputs):
        return inputs


def setup_decoder(task):
    return task.model(
        input_dim=task.model.input_dim,
        hidden_dim=task.model.hidden_dim,
        output_dim=task.model.output_dim,
        num_layers=task.model.num_layers
    )


def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)
    tasks_managers = initialize_tasks()
    train_dataloaders, val_dataloaders, test_dataloaders = prepare_dataloaders(tasks_managers)

    decoders = {task.name: setup_decoder(task) for task in tasks_managers}

    poi_mtl_model = Trainer(task_dict={task.name: task.evaluation_metrics for task in tasks_managers},
                            weighting=weighting_method.__dict__[params.weighting],
                            architecture=architecture_method.__dict__[params.arch],
                            encoder_class=Encoder,
                            decoders=decoders,
                            rep_grad=params.rep_grad,
                            multi_input=params.multi_input,
                            optim_param=optim_param,
                            scheduler_param=scheduler_param,
                            save_path=params.save_path,
                            load_path=params.load_path,
                            **kwargs)

    if params.mode == 'train':
        poi_mtl_model.train(train_dataloaders=train_dataloaders,
                            val_dataloaders=val_dataloaders,
                            test_dataloaders=test_dataloaders,
                            epochs=params.epochs)
    elif params.mode == 'test':
        poi_mtl_model.test(test_dataloaders)
    else:
        raise ValueError("Invalid mode selected. Choose 'train' or 'test'.")


if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    set_device(params.gpu_id)
    set_random_seed(params.seed)
    main(params)
