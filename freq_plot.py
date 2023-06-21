import random
import matplotlib.pyplot as plt
import pickle 
import os
import json
import pandas as pd
from collections import Counter
import pdb

if os.path.exists("../data/pico_dict.pickle"):
    with open("../data/pico_dict.pickle", "rb") as f:
        dataset = pickle.load(f)

random.seed(50)
# Shuffle and split the dataset into train_data, valid_data, and test_data
class_list = list(dataset.keys())
# strings_to_remove = ['iv-cont-q1', 'cv-cont-q1', 'iv-cont-q3', 'cv-cont-q3']
# class_list = list(filter(lambda x: x not in strings_to_remove, class_list))
random.shuffle(class_list)
num_classes = len(class_list)

train_classes = class_list[:num_classes // 2]
valid_classes = class_list[num_classes // 2:(num_classes * 3) // 4]
test_classes = class_list[(num_classes * 3) // 4:]

train_data = {k: dataset[k] for k in train_classes}
valid_data = {k: dataset[k] for k in valid_classes}
test_data = {k: dataset[k] for k in test_classes}

# Count the number of annotations in each data set
train_counts = {k: len(v) for k, v in train_data.items()}
valid_counts = {k: len(v) for k, v in valid_data.items()}
test_counts = {k: len(v) for k, v in test_data.items()}

counts_dict = {key: len(annotations) for key, annotations in dataset.items()}

def plot_freq_5_5():
    # Define the JSONL files
    jsonl_files = {
        'Train': '/work3/s174450/data/pico-episode-data/inter/pico_5_5_train_50.jsonl',
        'Dev': '/work3/s174450/data/pico-episode-data/inter/pico_5_5_dev_50.jsonl',
        'Test': '/work3/s174450/data/pico-episode-data/inter/pico_5_5_test_50.jsonl'
    }

    counts_dict = {
        'Train': train_counts,
        'Dev': valid_counts,
        'Test': test_counts
    }
    # Reverse the domain mapping to create an entity-domain mapping
    entity_domain_mapping = {entity: domain for domain, entities in domains.items() for entity in entities}
    # Initialize a dictionary for storing the domain counts
    domain_counts = {domain: [0, 0, 0] for domain in domains}
    # Parse the JSONL files
    for dataset, dict_counts in counts_dict .items():
        for key, item in dict_counts.items():
            if key in entity_domain_mapping:
                domain_counts[entity_domain_mapping[key]][list(counts_dict.keys()).index(dataset)] += item
    # for dataset, jsonl_file in jsonl_files.items():
    #     with open(jsonl_file, 'r') as file:
    #         for line in file:
    #             # Parse the episode
    #             episode = json.loads(line)
    #             # Count the instances of each domain
    #             for entity in episode['types']:
    #                 if entity in entity_domain_mapping:
    #                     domain_counts[entity_domain_mapping[entity]][list(jsonl_files.keys()).index(dataset)] += 10
    # Convert the domain count dictionary to a DataFrame
    pdb.set_trace()
    df = pd.DataFrame.from_dict(domain_counts, orient='index', columns=jsonl_files.keys())
    df = df.drop("O", errors='ignore')
    # Create the bar plot
    df.plot(kind='bar', stacked=False, figsize=(10, 7), rot=0)

    # Set the title and labels
    plt.title("Fine-grained Counts in Different Datasets for 5-Way 5-Shots")
    plt.xlabel("")
    plt.ylabel("Number of unique annotations")

    # Show the plot
    plt.savefig('images/barplot_5_1_50.png')


def plot_train_dev_test_freq():
    """
    Plots the frequencies of annotations in the training, validation, and test datasets.

    The function takes three dictionaries as input: train_counts, valid_counts, and test_counts.
    Each dictionary contains keys representing different entities from the PICO dataset, and the
    values are the frequencies of annotations for each entity.

    The function generates three bar charts, one for each dataset, showing the frequencies of
    annotations for each entity.

    The resulting plots are saved as an image in the 'images' directory.

    Parameters:
        train_counts (dict): Dictionary containing frequencies of annotations in the training dataset.
        valid_counts (dict): Dictionary containing frequencies of annotations in the validation dataset.
        test_counts (dict): Dictionary containing frequencies of annotations in the test dataset.

    Returns:
        None
    """
    # Plot the counts of annotations in each data set
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    ax[0].bar(train_counts.keys(), train_counts.values())
    ax[0].set_title('Training Data')
    ax[0].set_ylabel('Frequency')
    ax[0].set_ylim([0, 1100])
    ax[0].set_xticklabels(train_counts.keys(), rotation=45, ha="right")  # Rotate x-axis labels

    ax[1].bar(valid_counts.keys(), valid_counts.values())
    ax[1].set_title('Validation Data')
    ax[1].set_ylabel('Frequency')
    ax[1].set_ylim([0, 1100])

    ax[2].bar(test_counts.keys(), test_counts.values())
    ax[2].set_title('Test Data')
    ax[2].set_ylabel('Frequency')
    ax[2].set_ylim([0, 1100])

    plt.tight_layout()
    plt.savefig('images/Freq.png')

def plot_all_freq():
    """
    Plots a bar chart showing the number of text+annotations are available for each entity in PICO.

    The resulting plot is saved as an image in the 'images' directory.

    Parameters:
        counts_dict (dict): PICO data set dictionary with all entities.

    Returns:
        None
    """
    keys = counts_dict.keys()
    values = counts_dict.values()

    plt.figure(figsize=(12, 6))
    plt.bar(keys, values)
    plt.xticks(rotation=90)
    plt.xlabel('Keys')
    plt.ylabel('Number of Items')
    plt.title('Number of Items in Each Key')
    plt.tight_layout()
    plt.savefig('images/all_freq.png')

def plot_text_length():
    """
    Plots histograms of tokenized text lengths for training, validation, and testing datasets.

    The function extracts the text lengths from the provided data dictionaries and generates histograms
    to visualize the distribution of text lengths.

    The resulting histograms are saved as images in the 'images' directory.

    Parameters:
        train_data (dict): Dictionary containing training data.
        valid_data (dict): Dictionary containing validation data.
        test_data (dict): Dictionary containing testing data.

    Returns:
        None
    """
    def extract_text_lengths(data):
        text_lengths = []
        for tuples_list in data.values():
            for item in tuples_list:
                for text, annotation in item:
                    text_lengths.append(len(text))
        return text_lengths

    text_lengths_train = extract_text_lengths(train_data)
    text_lengths_dev = extract_text_lengths(valid_data)
    text_lengths_test = extract_text_lengths(test_data)

    def plot_histogram(text_lengths, title, file_type):
        plt.figure()
        plt.hist(text_lengths, bins=25, edgecolor='black')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.savefig(f'images/sequence_length_hist_{file_type}.png')

    plot_histogram(text_lengths_train, 'Histogram of Tokenized Text Lengths for Training', 'train')
    plot_histogram(text_lengths_dev, 'Histogram of Tokenized Text Lengths for Validation', 'dev')
    plot_histogram(text_lengths_test, 'Histogram of Tokenized Text Lengths for Testing', 'test')

def plot_pico_freq_jsonl():
    """
    Plots the frequencies of annotation types in three JSONL files: pico_5_1_train.jsonl,
    pico_5_1_dev.jsonl, and pico_5_1_test.jsonl.

    The function reads the JSONL files, extracts the annotation types, counts their occurrences,
    and generates bar plots showing the frequencies for each file.

    The resulting plots are saved as images in the 'images' directory.

    Parameters:
        None

    Returns:
        None
    """
    # Step 1: Read the JSONL files and parse them into dictionaries
    data_train, data_dev, data_test = [], [], []

    with open("data/pico-episode-data/pico_5_1_train.jsonl", "r") as file:
        for line in file:
            data_train.append(json.loads(line))

    with open("data/pico-episode-data/pico_5_1_dev.jsonl", "r") as file:
        for line in file:
            data_dev.append(json.loads(line))

    with open("data/pico-episode-data/pico_5_1_test.jsonl", "r") as file:
        for line in file:
            data_test.append(json.loads(line))

    # Step 2: Extract the annotation types
    def extract_annotation_types(data):
        annotation_types = []
        for entry in data:
            annotation_types.extend(entry["types"])
        return annotation_types

    annotation_types_train = extract_annotation_types(data_train)
    annotation_types_dev = extract_annotation_types(data_dev)
    annotation_types_test = extract_annotation_types(data_test)
    pdb.set_trace()
    # Step 3: Count the occurrences of each annotation type
    type_counts_train = Counter(annotation_types_train)
    type_counts_dev = Counter(annotation_types_dev)
    type_counts_test = Counter(annotation_types_test)

    # Step 4: Plot the annotation types and their frequencies
    def plot_annotation_types(type_counts, title, type):
        types = type_counts.keys()
        counts = type_counts.values()
        plt.figure()
        plt.bar(types, counts)
        plt.xlabel("Annotation Types")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'images/pico_freq_jsonl_{type}.png')

    plot_annotation_types(type_counts_train, "Annotation Type Frequencies (Train)", 'train')
    plot_annotation_types(type_counts_dev, "Annotation Type Frequencies (Dev)", 'dev')
    plot_annotation_types(type_counts_test, "Annotation Type Frequencies (Test)", 'test')



if __name__ == '__main__':
    domains = json.load(open("/work3/s174450/data/entity_types_pico.json"))
    plot_freq_5_5()
