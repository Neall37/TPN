import pandas as pd
import h5py
import torch
from ESM2 import get_embedding
from Bio import Entrez, SeqIO
import numpy as np
import csv
import time


def fetch_protein_sequences(species_name, max_sequences=22000, output_file='output_sequences.fasta'):
    query = f"{species_name}[ORGN]"
    with Entrez.esearch(db="ipg", term=query, retmax=max_sequences) as handle:
        search_results = Entrez.read(handle)
    #print(search_results)sequence_ids[1]
    sequence_ids = search_results['IdList']
    print(len(sequence_ids))
    chunk_size = 9999  # The maximum number of IDs per single request
    protein_accession_list = []

    # Break down sequence_ids into chunks of size chunk_size
    for j in range(0, len(sequence_ids), chunk_size):
        chunk = sequence_ids[j:j + chunk_size]
        handle = Entrez.esummary(db="ipg", id=','.join(chunk))
        summary = Entrez.read(handle)
        handle.close()

        # Extract information from each record in this chunk
        for record in summary['DocumentSummarySet']['DocumentSummary']:
            protein_accession_list.append(record['Accession'])

        time.sleep(0.25)
    sequences = fetch_sequences_in_batches(protein_accession_list)
    SeqIO.write(sequences, output_file, "fasta")

def fetch_sequences_in_batches(sequence_ids, batch_size=9999):
    sequences = []
    for i in range(0, len(sequence_ids), batch_size):
        time.sleep(0.2)
        batch_ids = sequence_ids[i:i+batch_size]
        handle = Entrez.efetch(db="protein", id=batch_ids, rettype="fasta", retmode="text")
        batch_sequences = list(SeqIO.parse(handle, "fasta"))
        handle.close()
        sequences.extend(batch_sequences)
    return sequences


def process_species_data(df, h5_name, uni_length=20000, datatype='labeled',
                        label_to_index_1 = {'hypha': 0, 'non_mycelium': 1},
                        label_to_index_2 = {'Saprotrophs': 0, 'parasite': 1, 'Symbionts': 2, 'Pathogens': 3}):
    positions_idx = []
    
    # Initialize HDF5 file for storage
    with h5py.File(h5_name, 'w') as hdf:
        hdf.create_dataset('sequences', shape=(0, 480, uni_length), maxshape=(None, 480, uni_length), dtype='float32')
        if datatype == 'labeled':
            # Define a compound data type with two integers
            dt = np.dtype([('label1', np.int32), ('label2', np.int32)])
            # Create a dataset with this compound data type, initially empty but resizable
            hdf.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=dt)
        for index, row in df.iterrows():
            position_idx = (h5_name, index)
            positions_idx.append(position_idx)

            species_name = row['Species']
            if datatype == 'labeled':
                label1 = row['Growth_form_template_y']
                label2 = row['primary_lifestyle_y']
                label1_as_number = label_to_index_1[label1]
                label2_as_number = label_to_index_2[label2]

            try:
                # Fetch protein sequences for the species
                output_file = f"Protein_{species_name}.fasta"
                fetch_protein_sequences(species_name, output_file=output_file)

                # Process sequences to get a tensor
                sequence, label_order = get_embedding(model_location="esm2_t12_35M_UR50D", fasta_file=output_file,
                              output_dir=None, species=species_name, target_length=uni_length)
                df_label = pd.DataFrame(label_order, columns=['protein_labels'])
                df_label.to_csv(f'Protein_labels_{species_name}.csv')
                # Convert tensor to numpy for HDF5 storage
                tensor_cpu = sequence.cpu()  # Move the tensor to CPU
                tensor_np = tensor_cpu.numpy()

                # Resize datasets to accommodate new data
                current_size = hdf['sequences'].shape[0]
                new_size = current_size + 1  # Increase size by one for the new entry

                hdf['sequences'].resize(new_size, axis=0)
                # Add the new data
                hdf['sequences'][current_size, :, :] = tensor_np
                if datatype == 'labeled':
                    hdf['labels'].resize(new_size, axis=0)
                    hdf['labels'][current_size] = (label1_as_number, label2_as_number)

                print(f"Processed {species_name}")
                if index % 5 == 0:
                    hdf.flush()
            except Exception as e:
                print(f"Error processing {species_name}: {e}")
                continue
    return positions_idx


if __name__ == "__main__":
    # Prepare your csv file containing species name, protein count and labels
    df = pd.read_csv('labeled.csv')
    step_size = 3400  # 5 files
    idx = 0
    for i in range(5000, 22000, step_size):
        ave = i+step_size
        # Filter DataFrame for ProteinCount between X and X+step_size
        filtered_df = df[(df['Protein_count'] >= i) & (df['Protein_count'] <= i+step_size)]
        # please change you label mapping accordingly
        positions_idx = process_species_data(df=filtered_df, uni_length=ave,
                                             h5_name=f"labeled_train_{i}.h5", datatype='labeled',
                                             label_to_index_1 = {'hypha': 0, 'non_mycelium': 1},
                                             label_to_index_2 = {'Saprotrophs': 0, 'parasite': 1, 'Symbionts': 2, 'Pathogens': 3})







