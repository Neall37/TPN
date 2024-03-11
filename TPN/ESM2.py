import torch
from esm import FastaBatchedDataset, pretrained, MSATransformer
from typing import Sequence, Tuple
from collections import defaultdict
import re
import torch.nn.functional as F
import pandas as pd
import numpy as np
import h5py


class BatchConverter(object):

    def __init__(self, alphabet, compact_seq_length: int = None):
        self.alphabet = alphabet
        self.compact_seq_length = compact_seq_length

    def segmentation(self, seq_encoded):
        segment_length = self.compact_seq_length // 2
        step_size = self.compact_seq_length // 4
        return [seq_encoded[i:i + segment_length] for i in range(0, len(seq_encoded), step_size)]

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        batch_labels_new = ()
        seq_str_list_new = ()
        seq_encoded_list_new = ()

        if self.compact_seq_length:
            for i, (label, seq_str, seq_encoded) in enumerate(
                    zip(batch_labels, seq_str_list, seq_encoded_list)
            ):
                # Applying sliding window to segment long sequence
                if len(seq_encoded) > self.compact_seq_length:
                    segments_seq = self.segmentation(seq_encoded)
                    for segment_seq in segments_seq:
                        batch_labels_new = (*batch_labels_new, label)
                        seq_str_list_new = (*seq_str_list_new, seq_str)
                        seq_encoded_list_new = (*seq_encoded_list_new, segment_seq)
                else:
                    batch_labels_new = (*batch_labels_new, label)
                    seq_str_list_new = (*seq_str_list_new, seq_str)
                    seq_encoded_list_new = (*seq_encoded_list_new, seq_encoded)
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list_new)
        # Batch size changes after segmentation
        batch_size = len(batch_labels_new)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
            device='cpu'
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels_new, seq_str_list_new, seq_encoded_list_new)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64, device='cpu')
            tokens[
                i,
                int(self.alphabet.prepend_bos): len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return labels, strs, tokens

def get_embedding(model_location: str, fasta_file, species, output_dir=None,
        toks_per_batch: int = 2000, repr_layers: list[int] = [-1],
        truncation_seq_length: int = 1000, target_length=20000, nogpu=False):

    # Load a pretrained model and its alphabet (set of symbols)
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()  # Set the model to evaluation mode

    # Check if the model is a Multiple Sequence Alignment (MSA) Transformer
    if isinstance(model, MSATransformer):
        raise ValueError("This script does not support MSA Transformer models.")

    # Use GPU for computation if available and not disabled by the user
    if torch.cuda.is_available() and not nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    # Load sequences from a FASTA file and create batches for processing
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=BatchConverter(alphabet, truncation_seq_length),
        batch_sampler=batches, num_workers=8
    )

    print(f"Read {fasta_file} with {len(dataset)} sequences")

    # Adjust layer indices for representation extraction
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]

    with torch.no_grad():  # Disable gradient computation for inference
        label_representations = []
        label_order = []
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            if torch.cuda.is_available() and not nogpu:
                toks = toks.to(device="cuda", non_blocking=True)
            #Perform inference
            out = model(toks, repr_layers=repr_layers)
            logits = out["logits"].to(device="cpu")
            # Get result to cpu for processing
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            # Process each sequence in the batch
            final_layer_key = max(representations.keys())
            accumulated_embeddings = defaultdict(list)
            for i, label in enumerate(labels):
                label_order.append(label)
                for layer, t in representations.items():
                    sequence_tensor = t[i].mean(0)

                    # Check if the tensor is already on the correct device and is a float tensor
                    sequence_tensor = sequence_tensor.float()

                    # Accumulate the embeddings for the final layer
                    accumulated_embeddings[label].append(sequence_tensor)

            # Now outside the loop, compute the mean representation for each label
            mean_representations = {
                label: torch.stack(embeddings).mean(dim=0)
                for label, embeddings in accumulated_embeddings.items()
            }

            # Create a list of representative vectors
            representative_vectors = [mean_representations[label] for label in mean_representations]
            representative_tensor = torch.stack(representative_vectors)
            label_representations.append(representative_tensor)

        # Move all tensors in the list to CUDA and convert them to float
        label_representations = [t.to('cuda').float() for t in label_representations]

        combined_tensor = torch.cat(label_representations, dim=0)

        combined_tensor_t = combined_tensor.transpose(0, 1)
        current_length = combined_tensor_t.shape[-1]
        pad_total = target_length - current_length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        adjusted_tensor = F.pad(combined_tensor_t, (pad_left, pad_right), 'constant', 0)

        return adjusted_tensor, label_order


if __name__ == "__main__":
    h5_name = "C:\\Origin\\Research\\iid\\TBN_code\\eval\\Verticillium_eval_esm2_t36_3B_UR50D.h5"
    uni_length = 100
    datatype = 'unlabeled'
    positions_idx = []
    # Update these mappings to match your labels
    label_to_index_1 = {'hypha': 0, 'non_mycelium': 1}
    label_to_index_2 = {'Saprotrophs': 0, 'parasite': 1, 'Symbionts': 2, 'Pathogens': 3}
    fasta_files = ["G:\\我的云端硬盘\\TBN\\eval_data\\csv\\UMAP\\seqdump.txt",
                   "G:\\我的云端硬盘\\TBN\\eval_data\\csv\\UMAP\\seqdump (1).txt",
                   "G:\\我的云端硬盘\\TBN\\eval_data\\csv\\UMAP\\seqdump (2).txt"
                   ]
    species_name = ["Verticillium_alfalfae", "Verticillium_dahliae","Verticillium_nonalfalfae","human"]
    with h5py.File(h5_name, 'w') as hdf:
        hdf.create_dataset('sequences', shape=(0, 480, uni_length), maxshape=(None, 480, uni_length), dtype='float32')
        if datatype == 'labeled':
            # Define a compound data type with two integers
            dt = np.dtype([('label1', np.int32), ('label2', np.int32)])
            # Create a dataset with this compound data type, initially empty but resizable
            hdf.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=dt)
        for i,fasta_file in enumerate(fasta_files):
            try:
                # Process sequences to get a tensor,esm2_t12_35M_UR50D,esm2_t33_650M_UR50D,esm2_t36_3B_UR50D
                sequence, label_order = get_embedding(model_location="esm2_t12_35M_UR50D", fasta_file=fasta_file,
                              output_dir=None, species=species_name[i], target_length=uni_length)
                tensor_cpu = sequence.cpu()  # Move the tensor to CPU
                tensor_np = tensor_cpu.numpy()

                current_size = hdf['sequences'].shape[0]
                new_size = current_size + 1  # Increase size by one for the new entry

                hdf['sequences'].resize(new_size, axis=0)
                # Add the new data
                hdf['sequences'][current_size, :, :] = tensor_np

                print(f"Processed {species_name[i]}")
            except Exception as e:
                print(f"Error processing {species_name[i]}: {e}")
                continue
