import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
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


# class WeightLoss(nn.Module):
#     def __init__(self, sigma1=1.0, sigma2=1.0):
#         super(WeightLoss, self).__init__()
#         # If the sigmas are learnable parameters
#         self.sigma1 = nn.Parameter(torch.tensor([sigma1]))
#         self.sigma2 = nn.Parameter(torch.tensor([sigma2]))

#     def forward(self, L1, L2):
#         epsilon = 1e-8  # A small constant to prevent division by zero or log of zero
#         sigma1_squared = self.sigma1 ** 2 + epsilon
#         sigma2_squared = self.sigma2 ** 2 + epsilon
#         loss = 1 / sigma1_squared * L1 + 1 / sigma2_squared * L2 + torch.log(sigma1_squared) + torch.log(sigma2_squared)
#         return loss


class LargeHDF5Dataset_augment(Dataset):
    def __init__(self, hdf5_files, index, data_type='labeled'):
        """
        Initialize dataset.

        Args:
        hdf5_files (list of str): List of paths to the HDF5 files containing the dataset.
        index (list of tuples): List of tuples where each tuple is (file_name, idx) indicating
                                the file and index within the file for each data point.
        data_type (str): Type of data to load ('labeled' or 'unlabeled').
        """
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
        """
        Initialize dataset.

        Args:
        hdf5_files (list of str): List of paths to the HDF5 files containing the dataset.
        index (list of tuples): List of tuples where each tuple is (file_name, idx) indicating
                                the file and index within the file for each data point.
        data_type (str): Type of data to load ('labeled' or 'unlabeled').
        """
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
        """
        Apply label smoothing and calculate the cross-entropy loss.
        :param input: logits from the model (shape: [batch_size, num_classes])
        :param target: ground truth labels (shape: [batch_size])
        """
        batch_size, num_classes = input.size()
        # Create smoothed labels
        true_dist = input.data.clone()  # Initialize with the input tensor
        true_dist.fill_(self.smoothing / (num_classes - 1))  # Fill with the smoothing value
        # Scatter 1 - smoothing value to the true label indices
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)

        # Calculate log probabilities
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
    # #Assuming labels_bytes is a tuple of byte strings (label1, label2)
    # labels_str = [(label[0].decode('utf-8'), label[1].decode('utf-8')) for label in labels_origin]

    # # Update these mappings to match your labels
    # label_to_index_1 = {'hypha': 0, 'non_myceliu': 1}
    # label_to_index_2 = {'Saprotrophs': 0, 'parasite': 1, 'Symbionts': 2, 'Pathogens': 3}

    # # Convert string labels to indices for each category
    # index_labels_1 = [label_to_index_1[label[0]] for label in labels_str]
    # index_labels_2 = [label_to_index_2[label[1]] for label in labels_str]
    # If labels are numbers already
    # # Separate the tuples into two lists, one for each position
    # labels_1 = [label[0] for label in labels_origin]
    # labels_2 = [label[1] for label in labels_origin]

    # Convert the lists to PyTorch tensors
    labels_1 = [label[0] for label in labels_origin]
    labels_2 = [label[1] for label in labels_origin]

    # Convert the lists to PyTorch tensors
    labels_tensor_1 = torch.tensor(labels_1, dtype=torch.long)
    labels_tensor_2 = torch.tensor(labels_2, dtype=torch.long)
    # print('1',sequences_padded[0].shape)
    # print('2',sequences_padded[1].shape)

    return torch.stack(sequences_padded), labels_tensor_1, labels_tensor_2


# Function to evaluate the model on the validation set
def evaluate_model(model, datasets_loader, status=True):
    if status:
        model.eval()  # Set the model to evaluation mode
        loss_function_category_1_test = nn.CrossEntropyLoss()
        loss_function_category_2_test = nn.CrossEntropyLoss()
        # Set the model to evaluation mode
        epoch_loss_test = 0.0
        correct_predictions_category_1_test = 0
        correct_predictions_category_2_test = 0
        total_samples_category_1_test = 0
        total_samples_category_2_test = 0
        # We need to collect the true labels and predictions for each category
        true_labels_category_1 = []
        predictions_category_1 = []
        true_labels_category_2 = []
        predictions_category_2 = []

        with torch.no_grad():  # No need to track gradients during evaluation
            for batch in datasets_loader:
                # Unpack the batch
                sequences, targets_category_1, targets_category_2 = batch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # Move the batch data to the designated device
                sequences = sequences.to(device)
                targets_category_1 = targets_category_1.to(device)
                targets_category_2 = targets_category_2.to(device)
                outputs_category_1_test, outputs_category_2_test = model(
                    sequences)  # If model expects a 'head', specify it here

                # loss_test=combined_loss(targets_category_1,outputs_category_1_test,
                #                    targets_category_2, outputs_category_2_test)
                loss_category_1_test = loss_function_category_1_test(outputs_category_1_test, targets_category_1)
                loss_category_2_test = loss_function_category_2_test(outputs_category_2_test, targets_category_2)

                # # Combine losses if necessary
                # #loss_test = weight_loss(loss_category_1_test, loss_category_2_test)
                loss_test = loss_category_1_test + loss_category_2_test

                epoch_loss_test += loss_test.item()
                _, predicted_category_1_test = torch.max(outputs_category_1_test.data, 1)
                _, predicted_category_2_test = torch.max(outputs_category_2_test.data, 1)

                correct_predictions_category_1_test += (predicted_category_1_test == targets_category_1).sum().item()
                correct_predictions_category_2_test += (predicted_category_2_test == targets_category_2).sum().item()
                total_samples_category_1_test += targets_category_1.size(0)
                total_samples_category_2_test += targets_category_2.size(0)
                # Move predictions and labels to CPU and collect them
                true_labels_category_1.extend(targets_category_1.cpu().numpy())
                predictions_category_1.extend(predicted_category_1_test.cpu().numpy())
                true_labels_category_2.extend(targets_category_2.cpu().numpy())
                predictions_category_2.extend(predicted_category_2_test.cpu().numpy())

            # Now, calculate weighted F1 score for each category
            weighted_f1_category_1 = f1_score(true_labels_category_1, predictions_category_1, average='weighted')
            weighted_f1_category_2 = f1_score(true_labels_category_2, predictions_category_2, average='weighted')
            avg_loss_test = epoch_loss_test / len(datasets_loader)
            accuracy_category_1_test = correct_predictions_category_1_test / total_samples_category_1_test
            accuracy_category_2_test = correct_predictions_category_2_test / total_samples_category_2_test
    else:
        avg_loss_test = 'NA'
        accuracy_category_1_test = 'NA'
        accuracy_category_2_test = 'NA'
    return (avg_loss_test, accuracy_category_1_test, accuracy_category_2_test,
            weighted_f1_category_1, weighted_f1_category_2)


def training_model(index_file, index_file_test=None,
                   status=False):
    # Setup1080
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
    # Assume 'input_data' is your input tensor and 'targets' are your labels

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

    if status:
        # Read the CSV file
        df_test = pd.read_csv(index_file_test)
        # Convert the dataframe to a list of tuples
        index_test = list(df_test.itertuples(index=False, name=None))
        # base_dir = './'  # Current directory; adjust if your files are elsewhere
        # List of file names
        file_names_test = [f"{i[0]}" for i in index_test]
        # Initialize your dataset
        dataset_test = LargeHDF5Dataset(hdf5_files=file_names_test, index=index_test, data_type='labeled')
        test_loader = DataLoader(dataset_test, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, collate_fn=collate_batch)

    num_epochs = 400
    # Define according to your dataset size and batch size
    total_num_samples = len(index)
    steps_per_epoch = int(total_num_samples / batch_size)
    test_every_n_epochs = 1  # Define the frequency of validation and upgrade checkpoint
    # Directory to save checkpoints
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    '''#Check if the data is labeled or unlabeled" is_labeled = labels != -1'''
    # weight_loss = WeightLoss().to(device)
    for epoch_i in range(num_epochs):
        model.train()  # Set the model to train mode
        # Initialize metrics
        epoch_loss = 0.0
        correct_predictions_category_1 = 0
        correct_predictions_category_2 = 0
        total_samples_category_1 = 0
        total_samples_category_2 = 0
        # We need to collect the true labels and predictions for each category
        true_labels_category_1_train = []
        predictions_category_1_train = []
        true_labels_category_2_train = []
        predictions_category_2_train = []
        accumulation_steps = 1
        # Use enumerate to iterate through the DataLoader
        pbar = tqdm(enumerate(training_loader), total=len(training_loader),
                    desc=f'Epoch {epoch_i + 1}')
        for i, batch in pbar:
            # Unpack the batch
            sequences, targets_category_1, targets_category_2 = batch

            # Move the batch data to the designated device
            sequences = sequences.to(device)
            targets_category_1 = targets_category_1.to(device)
            targets_category_2 = targets_category_2.to(device)
            # global_step = epoch_i * steps_per_epoch + i
            # # Calculate the learning rate fraction based on warmup
            # lr_frac = min(1.0, (global_step + 1) / max(1.0, num_warmup_steps))
            # # Manually update the learning rate
            # for g in optimizer.param_groups:
            #     g['lr'] = target_learning_rate * lr_frac

            # Process the batch
            # optimizer.zero_grad()
            outputs_category_1, outputs_category_2 = model(sequences)  # If model expects a 'head', specify it here
            # loss=combined_loss(targets_category_1,outputs_category_1, targets_category_2, outputs_category_2)
            loss_category_1 = loss_function_category_1(outputs_category_1, targets_category_1)
            loss_category_2 = loss_function_category_2(outputs_category_2, targets_category_2)

            # # Combine losses if necessary
            # #loss = weight_loss(loss_category_1, loss_category_2)
            loss = loss_category_1 + loss_category_2
            loss = loss / accumulation_steps
            # Perform backpropagation and optimization based on the combined loss
            loss.backward()
            # optimizer.step()

            epoch_loss += loss.item()
            _, predicted_category_1 = torch.max(outputs_category_1.data, 1)
            _, predicted_category_2 = torch.max(outputs_category_2.data, 1)

            correct_predictions_category_1 += (predicted_category_1 == targets_category_1).sum().item()
            correct_predictions_category_2 += (predicted_category_2 == targets_category_2).sum().item()
            total_samples_category_1 += targets_category_1.size(0)
            total_samples_category_2 += targets_category_2.size(0)
            # Move predictions and labels to CPU and collect them
            true_labels_category_1_train.extend(targets_category_1.cpu().numpy())
            predictions_category_1_train.extend(predicted_category_1.cpu().numpy())
            true_labels_category_2_train.extend(targets_category_2.cpu().numpy())
            predictions_category_2_train.extend(predicted_category_2.cpu().numpy())

            if (i + 1) % accumulation_steps == 0 or (i + 1) == steps_per_epoch:
                optimizer.step()  # Update model weights
                optimizer.zero_grad()  # Clear gradients for the next set of accumulation steps

            # Update the progress bar appropriately
            pbar.update(1)
        # Step the scheduler
        scheduler.step()

        if epoch_i % test_every_n_epochs == 0:
            val_loss = 0
            if status:
                # After validate_every_n_epochs times of epoch, evaluate the model
                (avg_loss_test, accuracy_category_1_test, accuracy_category_2_test,
                 weighted_f1_category_1_test, weighted_f1_category_2_test) = evaluate_model(model, test_loader)
                val_loss = avg_loss_test
                print(f'Epoch {epoch_i + 1}, Test Loss: {avg_loss_test:.4f}, '
                      f'Test Accuracy Category 1: {accuracy_category_1_test:.4f},'
                      f'Test Accuracy Category 2: {accuracy_category_2_test:.4f},'
                      f'Test F1 Category 1: {weighted_f1_category_1_test:.4f},'
                      f'Test F1 Category 2: {weighted_f1_category_2_test:.4f}')
                # # 使用wandb.log 记录你想记录的指标
                wandb.log({
                    "Test Accuracy Category 1": accuracy_category_1_test,
                    "Test Accuracy Category 2": accuracy_category_2_test,
                    "Test avg_epoch_loss": avg_loss_test,
                    "Test F1 Category 1": weighted_f1_category_1_test,
                    "Test F1 Category 2": weighted_f1_category_2_test
                })
        if (epoch_i + 1) % 50 == 0:
            # Save model checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'TBNGELU_checkpoint_epoch_{epoch_i + 1}.pth')
            torch.save({
                'epoch': epoch_i + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                # Add any other relevant info here
            }, checkpoint_path)
            # Change lr
        weighted_f1_category_1_train = f1_score(true_labels_category_1_train, predictions_category_1_train,
                                                average='weighted')
        weighted_f1_category_2_train = f1_score(true_labels_category_2_train, predictions_category_2_train,
                                                average='weighted')
        avg_epoch_loss = accumulation_steps * epoch_loss / len(training_loader)
        accuracy_category_1 = correct_predictions_category_1 / total_samples_category_1
        accuracy_category_2 = correct_predictions_category_2 / total_samples_category_2

        print(
            f"Train Epoch {epoch_i + 1}/{num_epochs}, Train Average Loss: {avg_epoch_loss:.4f}, "
            f"Train Accuracy Category 1: {accuracy_category_1:.4f}, "
            f"Train Accuracy Category 2: {accuracy_category_2:.4f},"
            f"Train weighted_f1_category_1: {weighted_f1_category_1_train:.4f},"
            f"Train weighted_f1_category_2: {weighted_f1_category_2_train:.4f}"
        )
        # # 使用wandb.log 记录你想记录的指标
        wandb.log({
            "Train Accuracy Category 1": accuracy_category_1,
            "Train Accuracy Category 2": accuracy_category_2,
            "Train avg_epoch_loss": avg_epoch_loss,
            "Train F1 Category 1": weighted_f1_category_1_train,
            "Train F1 Category 2": weighted_f1_category_2_train
        })


if __name__ == "__main__":
    # Initialize the dataset

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="TBN",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.001,
            "architecture": "TBN",
            "dataset": "CIFAR-100",
            "epochs": 60,
        }
    )
    random.seed(42)
    training_model('/home/wuyou/Desk/TBN_code/Model/Data/Data_final_csv/idx_train_ssh_augmented.csv',
                   '/home/wuyou/Desk/TBN_code/Model/Data/Data_final_csv/idx_test_ssh.csv',
                   status=True)

    # index_file_test = 'test_test.csv',

    # print(data_list)
    #
    # training_model(data_list, label_list)


