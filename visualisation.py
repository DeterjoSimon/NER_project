import os
import json
from matplotlib_venn import venn3
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import random

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

def plot_f1_scores(f1_meta, f1_proto):
    pass

def plot_venn_classes(N, K, train, dev, set):
    plt.figure(figsize=(10,10))
    venn = venn3([train, dev, test], set_labels=('Train', 'Dev', 'Test'))
    plt.savefig(f'images/Venn_{N}_{K}.png')

def plot_heatmap(N,K):
    # Prepare a dictionary for our data
    data = {
    "Train": {},
    "Dev": {},
    "Test": {}
    }

    # Calculate entity counts for each domain for each dataset
    for domain, entities in domains.items():
        data["Train"][domain] = len(set(entities) & train)
        data["Dev"][domain] = len(set(entities) & dev)
        data["Test"][domain] = len(set(entities) & test)

    # Convert the data dictionary to a DataFrame
    df = pd.DataFrame(data)
    # Remove the row for "O" label
    df = df.drop("O", errors='ignore')

    # Create the matrix plot using seaborn's heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues", linewidths=.5)
    plt.savefig(f'images/Heatmap_{N}_{K}.png')

def plot_scatter_classes(N, K):

    # Function to get random coordinates within a circle (for cluster)
    def get_random_coords(radius):
        r = radius * (random.random() ** 0.5)
        theta = random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    # Define the centers and radii of the domain clusters
    centers = [(1, 0), (2, 1.5), (3, 0)]
    radii = [0.8, 0.8, 0.8]
    domain_names = list(domains.keys())
    domain_names.remove('O')

    # Create a DataFrame to store the scatter plot data
    data = {'x': [], 'y': [], 'entity': [], 'dataset': [], 'domain': []}

    # Generate the random coordinates for the entities in each domain for each dataset
    for domain, center, radius in zip(domain_names, centers, radii):
        for entity in domains[domain]:
            x, y = get_random_coords(radius)
            x += center[0]
            y += center[1]
            if entity in train:
                data['x'].append(x)
                data['y'].append(y)
                data['entity'].append(entity)
                data['dataset'].append('Train')
                data['domain'].append(domain)
            if entity in dev:
                data['x'].append(x)
                data['y'].append(y)
                data['entity'].append(entity)
                data['dataset'].append('Dev')
                data['domain'].append(domain)
            if entity in test:
                data['x'].append(x)
                data['y'].append(y)
                data['entity'].append(entity)
                data['dataset'].append('Test')
                data['domain'].append(domain)

    # Convert the data dictionary to a DataFrame
    df = pd.DataFrame(data)

    # Create the scatter plot
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='x', y='y', hue='dataset', data=df, s=100)

    plt.xlabel('')
    plt.ylabel('')
    plt.tick_params(axis='both', labelsize=0)   

    # Add labels for the entities
    for i in range(df.shape[0]):
        plt.text(x=df.x[i], y=df.y[i], s=df.entity[i], alpha=0.7, fontsize=11)

    # Draw the domain clusters
    for center, radius, domain in zip(centers, radii, domain_names):
        circle = Circle(center, radius, fill=False)
        plt.gca().add_patch(circle)
        plt.text(x=center[0], y=center[1], s=domain, fontsize=12, ha='center')

    # Set the legend and the title
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    plt.title("Entities in Different Domains and Datasets")

    # Show the plot
    plt.savefig(f'images/Scatter_classes_{N}_{K}.png')


if __name__ == '__main__':
    # Call the plot function for each file
    # files_to_plot = ['dev_5_1.jsonl', 'train_5_1.jsonl', 'test_5_1.jsonl']
    # files_to_plot = ['dev.jsonl', 'test.jsonl', 'train.jsonl']
    # for file_path in  files_to_plot:
    #     plot_sequence_length_histogram(os.path.join('data/pico-episode-data/', file_path))

    # DOMAIN VENN #
    domains = json.load(open("/work3/s174450/data/entity_types_pico.json"))
    # Define the sets of entities for each stage
    # For 7 and 5
    # train = set(['age', 'iv-cont-mean', 'intervention-participants', 'iv-cont-sd', 'cv-cont-median', 'iv-bin-percent', 'outcome-Measure', 'ethinicity'])
    # dev = set(['outcome', 'iv-cont-median', 'total-participants', 'cv-bin-percent', 'eligibility', 'control-participants', 'cv-cont-sd'])
    # test = set(['iv-bin-abs', 'control', 'intervention', 'cv-bin-abs', 'cv-cont-mean', 'location', 'condition'])
    # For 5 and 5
    train = set(['iv-cont-mean', 'cv-cont-sd', 'iv-cont-q3', 'total-participants', 'eligibility', 'cv-bin-percent', 'iv-bin-percent', 'ethinicity', 'outcome-Measure', 'age', 'cv-cont-q1', 'iv-cont-median', 'control-participants'])
    dev = set(['iv-cont-sd', 'cv-cont-median', 'intervention-participants', 'outcome', 'iv-cont-q1', 'iv-bin-abs'])
    test = set(['control', 'intervention', 'cv-cont-q3', 'cv-bin-abs', 'cv-cont-mean', 'location', 'condition'])
    # Flatten the entity domains into a single set of entities
    all_entities = set()
    for entity_list in domains.values():
        all_entities.update(entity_list)
    # # Verify the entities are in the domains
    # assert train.issubset(all_entities), "Some entities in 'train' are not in the domains"
    # assert dev.issubset(all_entities), "Some entities in 'dev' are not in the domains"
    # assert test.issubset(all_entities), "Some entities in 'test' are not in the domains"
    # plot_venn_classes(7,5,train, dev,set)
    # #             #
    plot_heatmap(5,5)
    plot_scatter_classes(5, 5)
    f1_meta = [13.8, 5]

