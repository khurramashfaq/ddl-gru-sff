import torch


class Args:
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = 'train'  # 'train' or 'test' modes
        self.lr = 0.0001  # Learning Rate.
        self.batch_size = 1  # Training batch size
        self.epochs = 251  # Number of epochs for training
        self.weight_decay = 0.0001  # Weight decay for optimizer
        self.experiment_name = '_'  # Experiment name
        self.beta1 = 0.9 # Beta1 for Adam Optimizer
        self.beta2 = 0.999 # Beta2 for Adam Optimizer

        self.hidden_dims = [128] * 3  # Hidden dimensions for GRU layers
        self.context_norm = "instance"  # Normalization type for context encoder
        self.n_downsample = 2  # Number of downsample operations
        self.n_gru_layers = 3  # Number of GRU layers in the model
        self.update_iters = 32  # Number of iterations for updating depth
        self.mixed_precision = True  # Enable mixed precision training
        self.n_focvols = 4  # Number of focus volumes extracted

        self.restore_ckpt = False

        self.dataset = 'FT'

    def update_from_data_config(self, data_config):
        for attr, value in data_config.__dict__.items():
            setattr(self, attr, value)


class Data_Config:
    def __init__(self, dataset):
        self.dataset = dataset

        if  self.dataset == 'FT':
            self.data_path = ".\data\FT"



        elif self.dataset == 'FoD':
            self.data_path = ".\data\FoD"


        elif self.dataset == '':
             pass




def get_config(dataset='FT'):
    args = Args()
    data_config = Data_Config(dataset)
    args.update_from_data_config(data_config)
    return args




if __name__ == "__main__":
    config = get_config()
    print(f"Current dataset: {config.dataset}")
    print(f"Number of images: {config.img_num}")
    print(f"Data path: {config.data_path}")
