from torch import nn
from torch.utils.data import DataLoader, Dataset
import h5py
import pandas as pd
from sklearn.metrics import f1_score,precision_score, recall_score, roc_auc_score, average_precision_score
import random
from BPNet import BPNet



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
        # Scatter 1 - smoothing value to the true label indices
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


import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import numpy as np


def evaluate_model(model, datasets_loader):
    model.eval()  # Set the model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize lists to store aggregated outputs and labels
    true_labels_category_1, predictions_category_1 = [], []
    true_labels_category_2, predictions_category_2 = [], []
    probabilities_category_1, probabilities_category_2 = [], []
    with torch.no_grad():  # No need to track gradients during evaluation
        for batch in datasets_loader:
            sequences, targets_category_1, targets_category_2 = batch
            sequences, targets_category_1, targets_category_2 = sequences.to(device), targets_category_1.to(
                device), targets_category_2.to(device)

            outputs_category_1_test, outputs_category_2_test = model(sequences)

            # Convert outputs to probabilities
            probs_category_1 = torch.softmax(outputs_category_1_test[:, :-1], dim=1)
            probs_category_2 = torch.softmax(outputs_category_2_test[:, :-1], dim=1)  # Adjust if necessary

            # Collect true labels and predictions
            true_labels_category_1.extend(targets_category_1.cpu().numpy())

            predictions_category_1.extend(torch.max(probs_category_1, 1)[1].cpu().numpy())

            true_labels_category_2.extend(targets_category_2.cpu().numpy())

            predictions_category_2.extend(torch.max(probs_category_2, 1)[1].cpu().numpy())

            # Aggregate probabilities for AUROC
            probabilities_category_1.extend(probs_category_1.cpu().numpy())
            probabilities_category_2.extend(probs_category_2.cpu().numpy())

    # Convert lists to numpy arrays
    true_labels_category_1, predictions_category_1 = np.array(true_labels_category_1), np.array(predictions_category_1)
    true_labels_category_2, predictions_category_2 = np.array(true_labels_category_2), np.array(predictions_category_2)

    probabilities_category_1, probabilities_category_2 = np.array(probabilities_category_1), np.array(
        probabilities_category_2)

    # Calculate metrics
    precision_category_1 = precision_score(true_labels_category_1, predictions_category_1, average='weighted')
    recall_category_1 = recall_score(true_labels_category_1, predictions_category_1, average='weighted')
    f1_category_1 = f1_score(true_labels_category_1, predictions_category_1, average='weighted')
    auroc_category_1 = roc_auc_score(true_labels_category_1, probabilities_category_1[:, 1])

    precision_category_2 = precision_score(true_labels_category_2, predictions_category_2, average='weighted')
    recall_category_2 = recall_score(true_labels_category_2, predictions_category_2, average='weighted')
    f1_category_2 = f1_score(true_labels_category_2, predictions_category_2, average='weighted')
    auroc_category_2 = roc_auc_score(true_labels_category_2, probabilities_category_2, multi_class='ovr',
                                     average='macro')

    # Print metrics
    print("Category 1 Metrics:")
    print(
        f"Precision: {precision_category_1}, Recall: {recall_category_1}, F1: {f1_category_1}, AUROC: {auroc_category_1}")

    print("\nCategory 2 Metrics:")
    print(
        f"Precision: {precision_category_2}, Recall: {recall_category_2}, F1: {f1_category_2}, AUROC: {auroc_category_2}")


if __name__ == "__main__":
    random.seed(42)
    # channels = 240
    # num_layers = 5
    # in_channels = 480
    # pooling_type = 'max'
    # target_percentage = 0.8
    # stack_layer_num = 3  # When change this, don't forget to change it in collate_batch
    # model = TraitProtNet(channels=channels, in_channels=in_channels, stack_layer_num=stack_layer_num,
    #                      layers_num=num_layers,
    #                      pooling_type=pooling_type, target_percentage=target_percentage)
    model = BPNet()
    # Path to your checkpoint file
    #checkpoint_path = 'TBNGELU2_25_checkpoint_epoch_250.pth'
    checkpoint_path = 'BPNet_checkpoint_epoch_100.pth'
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Load the state dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    batch_size = 32
    num_workers = 0
    index_file_test = "idx_test_ssh_1.csv"
    # Read the CSV file
    df_test = pd.read_csv(index_file_test, encoding='gbk')
    # Convert the dataframe to a list of tuples
    index_test = list(df_test.itertuples(index=False, name=None))
    # List of file names
    file_names_test = [f"{i[0]}" for i in index_test]
    # Initialize your dataset
    dataset_test = LargeHDF5Dataset(hdf5_files=file_names_test, index=index_test, data_type='labeled')
    test_loader = DataLoader(dataset_test, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers, collate_fn=collate_batch)
    
    evaluate_model(model, test_loader)


