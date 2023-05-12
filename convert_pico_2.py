import json
import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple
import sys
import pdb
import pickle
import spacy 
from tqdm import tqdm
from spacy.language import Language
from spacy.lang.en import English

nlp = spacy.load("en_core_web_sm")
random.seed(10)

# Add custom sentence segmentation rule
@Language.component("custom_segmenter")
def custom_segmenter(doc):
    for token in doc[:-1]:
        if token.text == '\n':
            doc[token.i + 1].is_sent_start = True
        elif token.text == '(' and token.i > 0:
            doc[token.i].is_sent_start = False
    return doc

def custom_tokenizer(nlp):
    infixes = nlp.Defaults.infixes + [
        r"(?<=[a-zA-Z])=(?=\d)",
        r"(?<=\D)≈(?=\d)",
        r"(?<=\d)/(?=\d)",
        r"(?<=\d)±(?=\d)",
        r"[\w]+|[^\w\s]",
    ]
    infix_re = spacy.util.compile_infix_regex(infixes)
    
    nlp.tokenizer.infix_finditer = infix_re.finditer
    return nlp.tokenizer

def read_ann_file(file_path: str) -> Dict[str, Tuple[int, int, str]]:
    entities = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            entities[parts[1].split()[0]] = (int(parts[1].split()[1]), int(parts[1].split()[2]), parts[2])
    return entities

def read_txt_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def process_pico_files(txt_file: str, ann_file: str) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    nlp = English()
    nlp.add_pipe("sentencizer")
    nlp.tokenizer = custom_tokenizer(nlp)
    # Add custom_segmenter to the pipeline
    tag_dictionary = defaultdict(list)
    if ann_file == 'data/pico-corpus/24606847.ann':
        pdb.set_trace()
    
    # Process abstract
    txt_content = read_txt_file(txt_file)
    # Process spanned annotation
    ann_content = read_ann_file(ann_file)

    doc = nlp(txt_content)
    sorted_ann_content = sorted(ann_content.items(), key=lambda x: x[1][0])
    idx = 0
    # pdb.set_trace()
    for sentence in doc.sents:
        sentence_start = sentence.start_char
        sentence_end = sentence.end_char
        words = [token.text for token in sentence]
        labels = ['O'] * len(words)
        assert len(words) == len(labels), "Not same length"
        # pdb.set_trace()
        while idx < len(ann_content) and sorted_ann_content[idx][1][0] <= sentence_end:
            # Loop through all indices that could be contained in this sentence
            # The first boolean checks all indices
            # And these span annotations have to be contained within the sentence
            # Thus <= sentence_end. 
            label, (start, end, span) = sorted_ann_content[idx]
            if start >= sentence_start: # If the annotation is after the sentence start.
                # pdb.set_trace()
                for i, word in enumerate(words): # Le prob est là, words est pas bien split
                    if start <= sentence[i].idx and sentence[i].idx < end:
                        # Set the label for the span
                        labels[i] = label
            label_key = next((lbl for lbl in labels if lbl != 'O'), None)
            if label_key:
                # If in the sentence an entity was spotted, add the sentence
                # and the spanned anotation corresponding to the entity 
                # And resume the research with the next spanned annotation
                # One sentence could have several entities. 
                tag_dictionary[label_key].append((words, labels))
                words = [token.text for token in sentence]
                labels = ['O'] * len(words)

            idx += 1
    a = 0
    assert len(ann_content) == len(tag_dictionary), f"Not all entities were collected. \
                                                    This happened with this annotation: \
                                                    {ann_file}"
    
    return tag_dictionary

def generate_episodes(dataset: Dict[str, List[Tuple[List[str], List[str]]]], N: int, K: int, amount):
    episodes = []

    # Filter the classes with at least 2 * N annotations
    available_classes = [cls for cls, annotations in dataset.items() if len(annotations) >= 2 * K]
    with tqdm(total=amount) as pbar:
        for _ in range(amount):
            # Randomly select N classes
            selected_classes = random.sample(available_classes, N)

            # Sample N annotations for support and query sets from each class
            support = {"word": [], "label": []}
            query = {"word": [], "label": []}
            valid_episode = True
            for cls in selected_classes:
                annotations = dataset[cls]

                # Check if there are enough annotations left to sample
                if len(annotations) >= 2*K:
                # Randomly sample annotations without replacement for support and query sets
                    selected_annotations = random.sample(annotations, 2 * K)
                    support_example = selected_annotations[:N]
                    query_example = selected_annotations[N:]
                    support_example = [item for sublist in support_example for item in sublist]
                    query_example = [item for sublist in query_example for item in sublist]
                else:
                    valid_episode = False
                    available_classes.remove(cls)
                    pbar.set_postfix({"Elements Left": len(available_classes)})
                    pbar.update(1)
                    break
                
                if not valid_episode:
                    continue
            # Create an episode and append it to the episodes list
                try:
                    for words, labels in support_example:
                        support["word"].append(words)
                        support["label"].append(labels)
                    for words, labels in query_example:
                        query["word"].append(words)
                        query["label"].append(labels)
                except:
                    a = 0
            pbar.update(1)
            episodes.append({"support": support, "query": query, "types": selected_classes})

    return episodes



def split_dataset(data_path: str, N: int, K: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    
    Args:
    data_path: str, represents the data path where the PICO-corpus is.
    N: int, Number of classes per episode.
    K: int, Number of samples from each class to train on.
    """
    dataset = defaultdict(list) # Dictionary of all the classes, for each class 
    # you have a list of sequences containing an abstract and a span annotation. 
    # FIXME: The data set has different kind of classes. 
    if os.path.exists("data/pico_dict.pickle"):
        with open("data/pico_dict.pickle", "rb") as f:
            dataset = pickle.load(f)
    else:
        for file in tqdm(os.listdir(data_path), desc="Processing files"): # We process each text and ann file 
            # There are 2022 files - 1011 annotations and 1011 abstracts. 
            if file.endswith(".txt"):
                file_base = file[:-4]
                txt_file = os.path.join(data_path, file)
                ann_file = os.path.join(data_path, f"{file_base}.ann")
                # Extract the words and labels
                tag_dictionary = process_pico_files(txt_file, ann_file) 
                for key in tag_dictionary.keys():
                    if key in dataset:
                        dataset[key].append(tag_dictionary[key])
                    else:
                        dataset[key] = [tag_dictionary[key]]
        with open("data/pico_dict.pickle", "wb") as f:
            pickle.dump(dataset, f)


    
    
    pdb.set_trace()
    class_list = list(dataset.keys()) # ['T1', 'T2', 'T3' ..., 'T26']
    random.shuffle(class_list)
    num_classes = len(class_list)

    train_classes = class_list[:num_classes // 2] # 13
    valid_classes = class_list[num_classes // 2:(num_classes * 3) // 4] # 6
    test_classes = class_list[(num_classes * 3) // 4:] # 7

    train_data = {k: dataset[k] for k in train_classes} # Make a dictionary of each class
    valid_data = {k: dataset[k] for k in valid_classes}
    test_data = {k: dataset[k] for k in test_classes}

    train_episodes = generate_episodes(train_data, N, K, 20000)
    valid_episodes = generate_episodes(valid_data, N, K, 1000)
    test_episodes = generate_episodes(test_data, N, K, 5000)

    return train_episodes, valid_episodes, test_episodes

def save_jsonl(data: List[Dict], file_path: str):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    data_dir = "data/pico-corpus"
    N = 5
    K = 1
    train_episodes, valid_episodes, test_episodes = split_dataset(data_dir, N, K)
    save_jsonl(train_episodes, f'data/pico-episode-data/pico_{N}_{K}_train.jsonl')
    save_jsonl(valid_episodes, f'data/pico-episode-data/pico_{N}_{K}_dev.jsonl')
    save_jsonl(test_episodes, f'data/pico-episode-data/pico_{N}_{K}_test.jsonl')
