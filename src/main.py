import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SentenceGetter import SentenceGetter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from visualization.plotter import Plotter

if __name__ == "__main__":
    data = pd.read_csv("../data/GMB/ner_dataset.csv", encoding='latin1')
    data = data.drop(['POS'], axis=1) # No need for parts of speech tags
    data = data.fillna(method="ffill")
    # print(data.tail(10))

    getter = SentenceGetter(data)
    sentence = getter.get_next()
    # for i in sentence:
    #     print(i)

    plotter = Plotter()
    # Plotter.plot_histogram(getter.sentences, "Sentence length distribution")

    words = set(list(data['Word'].values))
    tags = list(set(data["Tag"].values))

    words2index = {w:i for i,w in enumerate(words)}
    tags2index = {t:i for i,t in enumerate(tags)}
    data['Word_idx'] = data['Word'].map(words2index)
    data['Tag_idx'] = data['Tag'].map(tags2index)

    # Groupby and collect columns
    data_group = data.groupby(['Sentence #'], as_index=False)[['Word', 'Tag', 'Word_idx', 'Tag_idx']].agg(lambda x: list(x))
    # Visualise data
    print(data_group.head())

    # In order to feed our sentences into a LSTM network, they all need to be the same size.
    # From the histogram we decided to set the length for all sentences to 50, and add an
    # "EOS" token for empty spaces. 
    max_len = 50
    X = [[w[0] for w in s] for s in getter.sentences]
    new_X = []
    for seq in X:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("EOS")
    new_X.append(new_seq)

    # Same for the entities, however we need to map our labels to numbers
    y = [[tags2index[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tags2index["O"])


    