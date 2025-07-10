#!/usr/bin/env python

"""
This file contains a set of methods for the online analysis of microendoscopic
one photon data using a "ring-CNN" background model.
"""

import numpy as np
import os
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader 

import caiman.base.movies
from caiman.paths import caiman_datadir

class CalciumDataset(Dataset):
    def __init__(self, files, batch_size=32, random_state=42, 
                 train=True, var_name_hdf5='mov', subindices=None):
        """ 
        Create a Dataset object for Ca datasets.
        Args:
            files: list of paths to HDF5 files
            batch_size: number of frames per batch
            random_state: RNG seed for shuffling
            train: whether to shuffle files each epoch
            var_name_hdf5: name of the variable in HDF5
            subindices: optional slicing of frame indices
        """
        if isinstance(files, str):
            files = [files]
        self.files = files
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_state)
        self.train = train
        self.var_name_hdf5 = var_name_hdf5

        dims, T = caiman.base.movies.get_file_size(files, var_name_hdf5=var_name_hdf5)
        self.dim = dims
        if subindices is not None:
            self.frame_indices = list(range(T))[subindices]
        else:
            self.frame_indices = list(range(T))

        self.total_frames = len(self.frame_indices)
        self.on_epoch_end()
    
    def __len__(self):
        return self.total_frames // self.batch_size

    def __getitem__(self, idx):
        start_idx = self.frame_indices(idx * self.batch_size)
        end_idx = start_idx + self.batch_size
        X = caiman.base.movies.load(self.files[0], subindices=slice(start_idx, end_idx),
                                    var_name_hdf5=self.var_name_hdf5)
        X = np.expand_dims(X.astype(np.float32), axis=1)
        X = torch.tensor(X)
        return X, X #return input-output pair for autoencoder-style training

    def on_epoch_end(self):
        if self.train:
            self.random_state.shuffle(self.files)

class MaskedConv2D(nn.Module):
    """ Creates a trainable ring convolutional kernel with non zero entries between
    user specified radius_min and radius_max. Uses a random uniform non-negative
    initializer unless specified otherwise.

    Args:
        output_dim: int, default: 1
            number of output channels (number of kernels)

        kernel_size: (int, int), default: (5, 5)
            dimension of 2d boundaing box

        strides: (int, int), default: (1, 1)
            stride for convolution (modifying that will downsample)

        radius_min: int, default: 2
            inner radius of kernel

        radius_max: int, default: 3
            outer radius of kernel (typically: 2*radius_max - 1 = kernel_size[0])

        initializer: 'uniform' initializer, default: 'uniform'
            initializer for ring weights. 'uniform' will choose from a non-negative
            random uniform distribution such that the expected value of the sum
            is 2.

        use_bias: bool, default: True
            add a bias term to each convolution kernel

    Returns:
        Masked_Conv2D: Pytorch layer
            A trainable layer implementing the convolution with a ring
    """
    def __init__(self, in_channels, out_channels, kernel_size=(5,5), stride=(1,1),
               radius_min=2, radius_max=3, initializer='uniform',
               use_bias=True): 
        super(MaskedConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.stride = stride
        self.use_bias = use_bias
        self.padding = (kernel_size[0] //2, kernel_size[1] // 2)

        xx = np.arange(-(kernel_size[0]-1)//2, (kernel_size[0]+1)//2)
        yy = np.arange(-(kernel_size[0]-1)//2, (kernel_size[0]+1)//2)
        [XX, YY] = np.meshgrid(xx, yy)
        R = np.sqrt(XX**2 + YY**2)
        R[(R < radius_min) | (R > radius_max)] = 0
        R[R>0] = 1

        self.register_buffer('mask', torch.from_numpy(R).float().unsqueeze(0).unsqueeze(0))
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.build_parameters(initializer)

    def build_parameters(self, initializer):
        # Initialize bias to zero
        if self.use_bias:
            nn.init.constant_(self.bias, 0)

        # Replicate the Keras initializers
        if initializer == 'uniform':
            nn.init.uniform_(self.weight, a=0, b=2 / self.mask.sum())
        elif initializer == 'he_normal': # he_normal
            nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        else:
            nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x):
        masked_weight = self.weight * self.mask 
        y = F.conv2d(x, masked_weight, self.bias, stride=self.stride, padding=self.padding)
        return y 

class Hadamard(nn.Module):
    """ Creates a PyTorch multiplicative layer that performs
    pointwise multiplication with a set of learnable weights, followed by
    a sum across channels. 
    """
    def __init__(self, channels, height, width, initializer_val=0.1):
        super(Hadamard, self).__init__()
        self.kernel = nn.Parameter(torch.empty(1, channels, height, width))
        nn.init.constant_(self.kernel, initializer_val)

    def forward(self, x):
        hm = x * self.kernel
        sm = torch.sum(hm, dim=1, keepdim=True)
        return sm

class Additive(nn.Module):
    """ Creates a PyTorch additive layer that performs
    pointwise addition with a set of learnable weights.

    Args:
        initializer: initializer_val, default: 0.0
    """
    def __init__(self, height, width, initializer_val=0.0):
        super(Additive, self).__init__()
        self.kernel = nn.Parameter(torch.empty(1, 1, height, width))
        nn.init.constant_(self.kernel, initializer_val)

    def forward(self, x):
        hm = torch.add(x, self.kernel)
        return hm

def cropped_loss(gSig=0):
    """ Returns a cropped loss function to exclude boundaries (not used)
    Args:
        gSig: int, default: 0
            number of pixels to crop from each boundary
    Returns:
        my_loss: cropped loss function
    """
    def my_loss(y_true, y_pred):
        if gSig > 0:
            error = torch.square(y_true[..., gSig:-gSig, gSig:-gSig] - y_pred[..., gSig:-gSig, gSig:-gSig])
        else:
            error = torch.square(y_true - y_pred) 
        return error
    return my_loss

def quantile_loss(qnt=.50):
    """ Returns a quantile loss function that can be used for training.
    Args:
        qnt: float, default: 0.5
            desired quantile (0 < qnt < 1)
    Returns:
        my_qnt_loss: quantile loss function
    """
    def my_qnt_loss(y_true, y_pred):
        error = y_true - y_pred 
        return torch.mean(torch.where(error > 0, error*qnt, error*(qnt - 1)))
    return my_qnt_loss

def total_variation_loss():
    """ Returns a total variation norm loss function that can be used for training.
    """
    def my_total_variation_loss(y_true, y_pred):
        error = y_true - y_pred
        tv_h = torch.sum(torch.abs(error[:, :, 1:, :] - error[:, :, :-1, :]))
        tv_w = torch.sum(torch.abs(error[:, :, :, 1:] - error[:, :, :, :-1]))
        return (tv_h + tv_w) / torch.numel(error)
    return my_total_variation_loss

def get_run_logdir():
    """ Returns the path to the directory where the model will be saved.
    The directory will be locates inside the caiman_data/my_logs.
    """
    root_logdir = os.path.join(caiman_datadir(), "my_logs")
    if not os.path.exists(root_logdir):
        os.mkdir(root_logdir)
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

class RingCNN_LN(nn.Module):
    """ PyTorch two-layer linear convolutional neural network with 
    ring shape convolutions
    """
    def __init__(self, shape, n_channels=2, gSig=5, r_factor=1.5,
                 use_add=True, initializer='uniform', width=5, use_bias=False):
        super().__init__()
        height, width_shape, in_channels = shape
        radius_min = int(gSig * r_factor)
        radius_max = radius_min + width
        ks = 2 * radius_max + 1 #Kernel size

        in_channels = shape[-1] 

        self.conv1 = MaskedConv2D(in_channels=in_channels, out_channels=n_channels,
                                  kernel_size=(ks, ks), radius_min=radius_min,
                                  radius_max=radius_max, initializer=initializer,
                                  use_bias=use_bias)                        
        self.hadamard = Hadamard(channels=n_channels, height=height, 
                                width=width_shape, initializer_val=0.1)
        self.use_add = use_add
        if self.use_add:
            self.add = Additive(height=height, width=width_shape, 
                                initializer_val=0.0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.hadamard(x)
        if self.use_add:
            x = self.add(x)
        return x 

class RingCNN_NL(nn.Module):
    """ PyTorch two-layer nonlinear convolutional neural network
    with ring shape convolutions.
    """
    def __init__(self, shape, n_channels=8, gSig=5, r_factor=1.5,
                 use_add=True, initializer='he_normal', width=5, activation='relu',
                 use_bias=True):
        super().__init__()
        height, width_shape, in_channels = shape
        radius_min = int(gSig * r_factor)
        radius_max = radius_min + width
        ks = 2 * radius_max + 1 #Kernel size

        in_channels_conv1 = shape[-1] 
        self.conv = MaskedConv2D(in_channels=in_channels_conv1, out_channels=n_channels,
                                  kernel_size=(ks, ks), radius_min=radius_min,
                                  radius_max=radius_max, initializer=initializer,
                                  use_bias=use_bias)
        
        self.activation = getattr(F, activation)
        # self.dense_layer = nn.Linear(in_features=n_channels, out_features=1, bias=(not use_add)) 
        #Keras Reshape->Dense->Reshape is equivalent to a 1x1 convolution
        self.final_conv = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, bias=(not use_add))
        self.use_add = use_add
        if use_add:
             self.add = Additive(height=height, width=width_shape, initializer_val=0.0)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        # b, c, h, w = x.shape
        # x_permuted = x.permute(0, 2, 3, 1) 
        # x_reshaped = x_permuted.reshape(b * h * w, c)
        # x_dense = self.dense_layer(x_reshaped)
        # x_out = x_dense.reshape(b, h, w, 1)
        # x = x_out.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        if self.use_add:
            x = self.add(x)
        return x

def rate_scheduler(factor=0.5, epoch_length=200, samples_length=1e4):
    """
    Returns a scheduler factory compatible with PyTorch.
    Decreases LR by `factor` every `nepochs = samples_length / epoch_length`.
    """
    # nepochs = samples_length / epoch_length
    nepochs = max(1, samples_length / epoch_length if epoch_length > 0 else 1)
    decay = factor ** (1 / nepochs)
    def scheduler_factory(optimizer):
        lr_lambda = lambda epoch: decay ** epoch 
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler_factory

def create_LN_model(Y=None, shape=None, n_channels=2, gSig=5, r_factor=1.5, 
                    use_add=True, initializer='uniform', width=5, use_bias=False):
    """ Creates a PyTorch convolutional neural network with ring shape convolutions
    and multiplicative layers.
    """
    if shape is None:
        raise ValueError("The 'shape' argument must be provided for the PyTorch model.")
    model_LN = RingCNN_LN(shape=shape, n_channels=n_channels, gSig=gSig,
                       r_factor=r_factor, use_add=use_add, initializer=initializer,
                       width=width, use_bias=use_bias)

    return model_LN

def create_NL_model(Y=None, shape=None, n_channels=8, gSig=5, r_factor=1.5,
                    use_add=True, initializer='he_normal',
                    activation='relu', width=5, use_bias=True):
    """ Creates a PyTorch convolutional neural network with ring shape convolutions 
    (no multiplicative layers)
    Args:
        Y: np.array, default: None
            dataset to be fit, used only if a percentile based initializer is
            used for the additive layer and can be left to None

        shape: tuple, default: (None, None, 1)
            dimensions of the FOV. Can be left to its default value

        n_channels: int, default: 2
            number of convolutional kernels

        gSig: int, default: 5
            radius of average neuron

        r_factor: float, default: 1.5
            expansion factor to determine inner radius

        width: int, default: 5
            width of ring kernel

        use_add: bool, default: True
            flag for using an additive layer

        initializer: 'uniform' or torch initializer, default: 'uniform'
            initializer for ring weights. 'uniform' will choose from a non-negative
            random uniform distribution such that the expected value of the sum
            is 2.

        lr: float, default: 1e-4
            (initial) learning rate

        pct: float, default: 10
            percentile used for initializing additive layer
 
        activation: str or torch initializer, default: 'relu'
            (nonlinear) activation function 

        loss: str or torch loss function
            loss function used for training

        use_bias: bool, default: False
            add a bias term to each convolution kernel
    Returns:
        model_NL: torch model compiled and ready to be trained.
        Optimizer: torch optimizer (updating each step)
        Criterion: torch loss function 
    """
    if shape is None:
        raise ValueError("The 'shape' argument must be provided for the PyTorch model.")
    model_NL = RingCNN_NL(shape=shape, n_channels=n_channels, gSig=gSig,
                       r_factor=r_factor, use_add=use_add, initializer=initializer,
                       activation=activation, width=width, use_bias=use_bias)
    return model_NL

def fit_model(model, Y, optimizer, criterion, patience=5, 
            val_split=0.2, batch_size=32, epochs=500, 
            schedule=None, device=None):
    """
    Fits the PyTorch Ring-CNN model with an interface similar to Keras.

    Args:
        model: PyTorch Ring-CNN model to be trained.
        Y (np.array): The dataset for training and validation.
        optimizer: A pre-configured PyTorch optimizer (e.g., torch.optim.Adam).
        criterion: A pre-configured loss function (e.g., nn.MSELoss()).
        patience (int): Patience for early stopping.
        val_split (float): Fraction of data for validation.
        batch_size (int): Batch size for training.
        epochs (int): Maximum number of epochs.
        schedule: Optional learning rate scheduler.
        device: The device to train on (CPU or CUDA).

    Returns:
        model: Trained model loaded with best weights.
        history: Dictionary containing training history.
        path_to_model: Path to the saved model weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 

    # Prepare data and DataLoaders
    if Y.ndim == 3:
        Y = np.expand_dims(Y, axis=-1)
    # Transpose to PyTorch format (N, C, H, W)
    Y_tensor = torch.from_numpy(Y.astype(np.float32)).permute(0, 3, 1, 2) 
    dataset = TensorDataset(Y_tensor, Y_tensor)
    
    # Manually split data for validation, similar to Keras' validation_split
    num_samples = len(dataset)
    val_size = int(val_split * num_samples)
    train_size = num_samples - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize learning rate scheduler if provided
    lr_scheduler = schedule(optimizer) if schedule else None

    # Setup for saving the best model, analogous to Keras' ModelCheckpoint
    run_logdir = get_run_logdir()
    os.makedirs(run_logdir, exist_ok=True)
    path_to_model = os.path.join(run_logdir, 'model.pt')

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    # Manual training loop (standard for PyTorch)
    print(f"Starting training on {device}")  
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        
        # Manual validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch_val, Y_batch_val in val_loader:
                X_batch_val, Y_batch_val = X_batch_val.to(device), Y_batch_val.to(device)
                outputs_val = model(X_batch_val)
                loss_val = criterion(outputs_val, Y_batch_val)
                val_losses.append(loss_val.item())

        avg_val_loss = np.mean(val_losses) 
       
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")

        # Manual implementation of EarlyStopping and saving the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), path_to_model)
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        if lr_scheduler:
            lr_scheduler.step()

    print(f"Loading best model weights from {path_to_model}")
    model.load_state_dict(torch.load(path_to_model, map_location=device))
    return model, history, path_to_model
