class SentenceGetter(object):
    """
    This class converts every sentence with its named entitie into a list
    of tuple. [(word), (entity), ...]
    """

    def __init__(self, data) -> None:
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(word, tag) for word, tag in zip(
            s["Word"].values.tolist(),
            s["Tag"].values.tolist()
        )]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped[f"Sentence: {self.n_sent}"]
            self.n_sent += 1
            return s
        except:
            return None
        