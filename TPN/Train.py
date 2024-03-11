import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from Modules import TraitProtNet
import os
from pathlib import Path
import wandb
import h5py
import pandas as pd
from sklearn.metrics import f1_score
import random
from sklearn.model_selection import train_test_split


class LargeHDF5Dataset_augment(Dataset):
    def __init__(self, hdf5_files, index, data_type='labeled'):
        self.hdf5_files = hdf5_files
        self.index = index
        self.data_type = data_type
        self.file_handlers = {file_name: h5py.File(file_name, 'r') for file_name in hdf5_files}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_name, data_idx = self.index[idx]
        file = self.file_handlers[file_name]

        if self.data_type == 'labeled':
            sequence = file['augmented_sequences'][data_idx]
            label = file['augmented_labels'][data_idx]
            return sequence, label
        else:
            sequence = file['augmented_sequences'][data_idx]
            return sequence

    def close(self):
        for file in self.file_handlers.values():
            file.close()


class LargeHDF5Dataset(Dataset):
    def __init__(self, hdf5_files, index, data_type='labeled'):
        self.hdf5_files = hdf5_files
        self.index = index
        self.data_type = data_type
        self.file_handlers = {file_name: h5py.File(file_name, 'r') for file_name in hdf5_files}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_name, data_idx = self.index[idx]
        file = self.file_handlers[file_name]

        if self.data_type == 'labeled':
            sequence = file['sequences'][data_idx]
            label = file['labels'][data_idx]
            return sequence, label
        else:
            sequence = file['sequences'][data_idx]
            return sequence

    def close(self):
        for file in self.file_handlers.values():
            file.close()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        batch_size, num_classes = input.size()
        # Create smoothed labels
        true_dist = input.data.clone()  # Initialize with the input tensor
        true_dist.fill_(self.smoothing / (num_classes - 1))  # Fill with the smoothing value
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(input, dim=-1)

        # Compute the cross-entropy loss
        loss = (-true_dist * log_probs).sum(dim=1).mean()
        return loss


def collate_batch(batch):
    sequences, labels_origin = zip(*batch)
    sequences = [torch.from_numpy(sequence) for sequence in sequences]
    stack_layer_num = 2
    divide = 2 ** (stack_layer_num + 1)
    max_length = max(seq.shape[1] for seq in sequences)
    max_length_adjusted = ((max_length + 7) // divide) * divide
    # Pad sequences to the maximum length

    sequences_padded = [F.pad(seq, (0, max_length_adjusted - seq.shape[1]), "constant", 0) for seq in sequences]

    # Convert the lists to PyTorch tensors
    labels_1 = [label[0] for label in labels_origin]
    labels_2 = [label[1] for label in labels_origin]

    # Convert the lists to PyTorch tensors
    labels_tensor_1 = torch.tensor(labels_1, dtype=torch.long)
    labels_tensor_2 = torch.tensor(labels_2, dtype=torch.long)

    return torch.stack(sequences_padded), labels_tensor_1, labels_tensor_2


def training_model(index_file, index_file_test=None):
    channels = 240
    num_layers = 5
    in_channels = 480
    pooling_type = 'max'
    target_percentage = 0.8
    stack_layer_num = 3  # When change this, don't forget to change it in collate_batch
    model = TraitProtNet(channels=channels, in_channels=in_channels, stack_layer_num=stack_layer_num,
                         layers_num=num_layers,
                         pooling_type=pooling_type, target_percentage=target_percentage)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    learning_rate = 0.0005  # Initial learning rate
    # target_learning_rate = 0.00005
    # num_warmup_steps = 1000
    batch_size = 32
    num_workers = 8
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    optimizer.zero_grad()  # Initialize gradient to zero
    # Define ReduceLROnPlateau scheduler LabelSmoothingCrossEntropy(smoothing=0.1)
    loss_function_category_1 = nn.CrossEntropyLoss()
    loss_function_category_2 = LabelSmoothingCrossEntropy(smoothing=0.1)
    # Base directory where your HDF5 files are stored
    # Read the CSV file
    df = pd.read_csv(index_file)
    # Convert the dataframe to a list of tuples
    index = list(df.itertuples(index=False, name=None))
    # base_dir = './'  # Current directory; adjust if your files are elsewhere
    # List of file names
    file_names = [f"{i[0]}" for i in index]
    # Initialize your dataset
    dataset_training = LargeHDF5Dataset_augment(hdf5_files=file_names, index=index, data_type='labeled')

    # DataLoader instances for datasets
    training_loader = DataLoader(dataset_training, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, collate_fn=collate_batch)
                       
    num_epochs = 400
    # Define according to your dataset size and batch size
    total_num_samples = len(index)
    steps_per_epoch = int(total_num_samples / batch_size)
    test_every_n_epochs = 1  # Define the frequency of validation and upgrade checkpoint
    # Directory to save checkpoints
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch_i in range(num_epochs):
        model.train()  # Set the model to train mode
        accumulation_steps = 1
        pbar = tqdm(enumerate(training_loader), total=len(training_loader),
                    desc=f'Epoch {epoch_i + 1}')
        for i, batch in pbar:
            # Unpack the batch
            sequences, targets_category_1, targets_category_2 = batch

            # Move the batch data to the designated device
            sequences = sequences.to(device)
            targets_category_1 = targets_category_1.to(device)
            targets_category_2 = targets_category_2.to(device)
            outputs_category_1, outputs_category_2 = model(sequences)  # If model expects a 'head', specify it here
            loss_category_1 = loss_function_category_1(outputs_category_1, targets_category_1)
            loss_category_2 = loss_function_category_2(outputs_category_2, targets_category_2)

            loss = loss_category_1 + loss_category_2
            loss = loss / accumulation_steps
            # Perform backpropagation and optimization based on the combined loss
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == steps_per_epoch:
                optimizer.step()  # Update model weights
                optimizer.zero_grad()  # Clear gradients for the next set of accumulation steps

            # Update the progress bar appropriately
            pbar.update(1)
        # Step the scheduler
        scheduler.step()

if __name__ == "__main__":
    random.seed(42)
    training_model('idx_train_ssh_augmented.csv',
                   'idx_test_ssh.csv')




