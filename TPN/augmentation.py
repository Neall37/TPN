import h5py
import numpy as np
import torch

def shuffle_order_torch(data):
    """
    Randomly shuffle the rows and columns of the data in PyTorch.
    """

    # Shuffle columns
    shuffled_col_indices = torch.randperm(data.size(1))
    if data.is_cuda:
        shuffled_col_indices = shuffled_col_indices.to('cuda')

    # Apply column shuffling
    data = data[:, shuffled_col_indices]

    return data


def random_deletion_torch(data, deletion_rate=0.1):
    """
    Randomly delete some columns (lengths) of the data in PyTorch.
    """
    print(data.shape)
    n_columns = data.size(1)  # Size of the second dimension
    n_delete = int(n_columns * deletion_rate)
    indices = torch.randperm(n_columns)[:n_columns - n_delete]
    # print(indices.shape)
    if data.is_cuda:
        indices = indices.to('cuda')
    # Use all columns but only selected columns
    augmented_data = data[:, indices]
    return augmented_data


def add_gaussian_noise_torch(sequence, noise_mean=0, noise_std=0.1):
    gaussian_noise = torch.normal(mean=noise_mean, std=noise_std, size=sequence.size())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gaussian_noise = gaussian_noise.to(device)
    noisy_sequence = sequence + gaussian_noise
    return noisy_sequence


def augment(sequences, batch_labels_tensor_1, batch_labels_tensor_2, augmentations_per_sequence=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequences = sequences.to(device)
    batch_labels_tensor_1 = batch_labels_tensor_1.to(device)
    batch_labels_tensor_2 = batch_labels_tensor_2.to(device)
    augmented_sequences = []
    augmented_labels_tensor_1 = []
    augmented_labels_tensor_2 = []
    for i, sequence in enumerate(sequences):

        # For each original sequence, create multiple augmented versions
        for _ in range(augmentations_per_sequence):
            # Apply a series of augmentations; you can vary these for more diversity
            augmented_sequence = shuffle_order_torch(sequence)
            print('sss', augmented_sequence.shape)
            augmented_sequence = random_deletion_torch(augmented_sequence)
            augmented_sequence = add_gaussian_noise_torch(augmented_sequence)
            # Append each augmented sequence to the list
            augmented_sequences.append(augmented_sequence)
            augmented_labels_tensor_1.append(batch_labels_tensor_1[i])
            augmented_labels_tensor_2.append(batch_labels_tensor_2[i])
    return torch.stack(augmented_sequences), torch.stack(augmented_labels_tensor_1), torch.stack(
        augmented_labels_tensor_2)


# Apply the scaler to each chunk of data and save the results
def augment_and_save_batch(file_paths, batch_size=16):
    for path in file_paths:
        with h5py.File(path, 'r+') as f:
            data_size = f['sequences'].shape[0]
            labels = f['labels'][:]
            initial_size = f['sequences'].shape[-1]
            deletion_rate = 0.1
            truncate_length = initial_size * (1 - deletion_rate)
            # Prepare storage for augmented data and labels if not already existing
            if 'augmented_sequences' not in f:
                aug_seq_ds = f.create_dataset('augmented_sequences',
                                              shape=(0, f['sequences'].shape[1], truncate_length),
                                              maxshape=(None, f['sequences'].shape[1], truncate_length), chunks=True)
            else:
                aug_seq_ds = f['augmented_sequences']
            if 'augmented_labels' not in f:
                dt = np.dtype([('label1', np.int32), ('label2', np.int32)])
                # Create a dataset with this compound data type, initially empty but resizable
                aug_labels_ds = f.create_dataset('augmented_labels', shape=(0,), maxshape=(None,), dtype=dt)
            else:
                aug_labels_ds = f['augmented_labels']

            for start_idx in range(0, data_size, batch_size):
                end_idx = min(start_idx + batch_size, data_size)
                batch_data = f['sequences'][start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]

                # Convert to PyTorch tensors
                batch_data_tensor = torch.tensor(batch_data, dtype=torch.float)
                batch_labels_tensor_1 = torch.tensor([label[0] for label in batch_labels], dtype=torch.long)
                batch_labels_tensor_2 = torch.tensor([label[1] for label in batch_labels], dtype=torch.long)
                print(batch_data_tensor.shape)
                # Perform augmentation on the batch (Assuming your augment function returns PyTorch tensors)
                augmented_data_tensor, augmented_labels_tensor_1, augmented_labels_tensor_2 = augment(batch_data_tensor,
                                                                                                      batch_labels_tensor_1,
                                                                                                      batch_labels_tensor_2)
                print(augmented_data_tensor.shape)
                augmented_data_tensor_cpu = augmented_data_tensor.cpu()
                labels_1_np = augmented_labels_tensor_1.cpu().numpy()  # Move to CPU, then convert
                labels_2_np = augmented_labels_tensor_2.cpu().numpy()  # Move to CPU, then convert

                # Convert back to NumPy arrays to store in HDF5
                augmented_data = augmented_data_tensor_cpu.numpy()
                paired_labels = [(labels_1_np[i], labels_2_np[i]) for i in range(labels_1_np.shape[0])]
                # Append augmented data and labels to the datasets
                new_size = aug_seq_ds.shape[0] + augmented_data.shape[
                    0]  # Calculate the new size for the first dimension
                aug_seq_ds.resize(new_size, axis=0)
                aug_seq_ds[-augmented_data.shape[0]:, :, :] = augmented_data

                aug_labels_ds.resize(aug_labels_ds.shape[0] + len(paired_labels), axis=0)
                aug_labels_ds[-len(paired_labels):] = paired_labels


file_paths = ['labeled_train_5000.h5',
              'labeled_train_8400.h5',
              'labeled_train_11800.h5',
              'labeled_train_15200.h5',
              'labeled_train_18600.h5']
augment_and_save_batch(file_paths)
