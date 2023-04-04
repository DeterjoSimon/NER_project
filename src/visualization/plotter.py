import matplotlib.pyplot as plt
plt.style.use("ggplot")

class Plotter():

    def __init__(self, nrows=1, ncols=1):
        self.nrows = nrows
        self.ncols = ncols

    def plot_histogram(data, plot_title:str):
        plt.hist([len(sen) for sen in data], bins = 50)
        plt.title(plot_title)
        plt.show()
