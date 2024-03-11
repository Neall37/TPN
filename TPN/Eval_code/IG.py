from captum.attr import IntegratedGradients
from Modules import TraitProtNet
import matplotlib.pyplot as plt
import h5py
import torch
import numpy as np


def adjust_positions(target_length, positions, current_length):
    target_length = target_length
    pad_total = target_length - current_length
    pad_left = pad_total // 2
    # Adjust positions to reflect their original indices in the unpadded sequence
    adjusted_positions = positions - pad_left
    # Ensure adjusted positions are within the original length and non-negative
    adjusted_positions = adjusted_positions[(adjusted_positions >= 0) & (adjusted_positions < current_length)]
    return adjusted_positions


channels = 240
num_layers = 5
in_channels = 480
pooling_type = 'max'
target_percentage = 0.8
stack_layer_num = 3
model = TraitProtNet(channels=channels, in_channels=in_channels, stack_layer_num=stack_layer_num,
                     layers_num=num_layers,
                     pooling_type=pooling_type, target_percentage=target_percentage)

# Path to your checkpoint file
checkpoint_path = 'C:\\Origin\\Research\\iid\\TBN_code\\Model\Data\\checkpoints\\TBNGELU2_25_checkpoint_epoch_250.pth'

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Load the state dict into the model
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

h5 = h5py.File("Verticillium_eval.h5", 'r')

input_tensors = h5['sequences']
species_name = ["Verticillium_alfalfae", "Verticillium_dahliae","Verticillium_nonalfalfae"]
length_list = [10246,10703,9441]

for i, input_tensor in enumerate(input_tensors):
    # labels = h5['labels'][i]
    # targets_category_1, targets_category_2 = labels
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.from_numpy(input_tensor)
    input_tensor = input_tensor.unsqueeze(0)
    current_length = length_list[i]
    print('current_length: ', current_length)
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # print(f'True_task1: {targets_category_1}, True_label_task2: {targets_category_2}')
    outputs_category_1_test, outputs_category_2_test = model(input_tensor)  # If model expects a 'head', specify it here

    _, predicted_category_1_test = torch.max(outputs_category_1_test.data, 1)
    _, predicted_category_2_test = torch.max(outputs_category_2_test.data, 1)
    print(f'Predict_task1:{predicted_category_1_test}, Predict_task2:{predicted_category_2_test}')


    def model_wrapper_x1(input_tensor):
        return model(input_tensor)[0]  # Only return x1


    def model_wrapper_x2(input_tensor):
        return model(input_tensor)[1]  # Only return x2


    ig1 = IntegratedGradients(model_wrapper_x1)

    # Compute attributions for the first task
    attributions_task1 = ig1.attribute(input_tensor, target=0)

    # For the second task, assuming model_wrapper_x2 is a different function or model instance
    ig2 = IntegratedGradients(model_wrapper_x2)

    # Compute attributions for the second task
    attributions_task2 = ig2.attribute(input_tensor, target=3)

    # Sum attributions across all features for each time step in the sequence
    attributions_sum_1 = attributions_task1.sum(dim=-2)
    attributions_sum_2 = attributions_task2.sum(dim=-2)

    # Assuming attributions_sum_1 and attributions_sum_2 are your attribution sums for each task
    # Detach and convert to numpy if they're still torch tensors
    attributions_sum_1_np = attributions_sum_1.detach().cpu().numpy().squeeze()
    attributions_sum_2_np = attributions_sum_2.detach().cpu().numpy().squeeze()

    # Threshold for selecting the high-attribution positions
    threshold = 0.0025

    # Get the indices/positions where the attributions are greater than the threshold for both tasks
    positions_above_threshold_task1 = np.where(attributions_sum_1_np >= threshold)[0]
    positions_above_threshold_task2 = np.where(attributions_sum_2_np >= threshold)[0]

    # Get the corresponding values for those positions for both tasks
    values_above_threshold_task1 = attributions_sum_1_np[positions_above_threshold_task1]
    values_above_threshold_task2 = attributions_sum_2_np[positions_above_threshold_task2]

    adjusted_positions_task1 = adjust_positions(target_length=11000, positions_above_threshold_task1, current_length)
    adjusted_positions_task2 = adjust_positions(target_length=11000,positions_above_threshold_task2, current_length)

    # Create a combined bubble plot
    # Assuming attributions_sum_1_np and attributions_sum_2_np are your data arrays
    plt.figure(figsize=(15, 8))
    time_steps = np.arange(len(attributions_sum_1_np))

    # Task 1
    positive_mask_1 = attributions_sum_1_np > 0
    plt.scatter(time_steps[positive_mask_1], attributions_sum_1_np[positive_mask_1],
                s=np.abs(attributions_sum_1_np[positive_mask_1]) * 1000, alpha=0.5, color='blue',
                marker='o', label='Task 1 Positive Attributions')
    plt.scatter(time_steps[~positive_mask_1], attributions_sum_1_np[~positive_mask_1],
                s=np.abs(attributions_sum_1_np[~positive_mask_1]) * 1000, alpha=0.5, color='red',
                marker='o', label='Task 1 Negative Attributions')

    # Task 2
    positive_mask_2 = attributions_sum_2_np > 0
    plt.scatter(time_steps[positive_mask_2], attributions_sum_2_np[positive_mask_2],
                s=np.abs(attributions_sum_2_np[positive_mask_2]) * 1000, alpha=0.5, color='green',
                marker='o', label='Task 2 Positive Attributions')
    plt.scatter(time_steps[~positive_mask_2], attributions_sum_2_np[~positive_mask_2],
                s=np.abs(attributions_sum_2_np[~positive_mask_2]) * 1000, alpha=0.5, color='#FFA500',
                marker='o', label='Task 2 Negative Attributions')

    plt.title('Combined Bubble Plot for Task 1 and Task 2')
    plt.xlabel('Time Step')
    plt.ylabel('Attribution Sum')
    plt.legend()
    plt.show()
