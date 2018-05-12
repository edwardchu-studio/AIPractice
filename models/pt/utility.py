import numpy as np
import matplotlib as plt

plt.interactive()

class Utility:
    __doc__ = "a utility class included multi functions"

    def __init__(self):
        pass

    def sigmoid(self,inputs):
        """
        Calculate the sigmoid for the give inputs (array)
        :param inputs:
        :return:
        """
        sigmoid_scores = [1 / float(1 + np.exp(- x)) for x in inputs]
        return sigmoid_scores
    def softmax(self,inputs):
        """
        Calculate the softmax for the give inputs (array)
        :param inputs:
        :return:
        """
        return np.exp(inputs) / float(sum(np.exp(inputs)))


    def line_graph(self,x, y, x_title, y_title):
        """
        Draw line graph with x and y values
        :param x:
        :param y:
        :param x_title:
        :param y_title:
        :return:
        """
        plt.plot(x, y)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.show()
