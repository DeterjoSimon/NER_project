import jsonlines
import matplotlib.pyplot as plt
import os
# Define a function that creates the plot for a single file
def plot_sequence_length_histogram(file_path):
    # Open the jsonl file and read in the episodes
    episodes = []
    with jsonlines.open(file_path) as reader:
        for episode in reader:
            episodes.append(episode)

    # Get the sequence lengths for each episode and flatten them into a single list
    sequence_lengths = []
    for episode in episodes:
        for sequence in episode['support']['word'] + episode['query']['word']:
            sequence_lengths.append(len(sequence))

    # Create a frequency plot of the sequence lengths
    plt.figure()
    plt.hist(sequence_lengths, bins=10)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title(f'Sequence Length Histogram: {file_path}')
    plt.savefig('images/sequence_length_histogram_' + os.path.basename(file_path).split(".")[0])

# Call the plot function for each file
# files_to_plot = ['dev_5_1.jsonl', 'train_5_1.jsonl', 'test_5_1.jsonl']
files_to_plot = ['dev.jsonl', 'test.jsonl', 'train.jsonl']
for file_path in  files_to_plot:
    plot_sequence_length_histogram(os.path.join('data/pico-episode-data/', file_path))
