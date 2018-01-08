import numpy as np


class Perceptron:
    """
        Perceptron is one of the first machine learning models
        it was proposed by F.Rosenblatt
    """
    def __init__(self, learning_rate=0.01, iterations=10):
        """
        Initializes the perceptron
        :param learning_rate: The factor that updates the weights
        :param iterations: total no of samples that the Perceptron has to look over
        to update its weights
        :return:
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    def fit(self, data, real_output):
        """
        This method updates the perceptron's weights
        :param data: The input data, which is used by perceptron to update its weights
        it is a 2-D numpy array with each row being a new input
        :param real_output: The output for each input
        :return:
        """
        self.weights = np.zeros(data.shape[1] + 1)  # shape[1]->no of features
        # adding 1 to an independent weight
        for i in range(0, self.iterations):
            for input_given, output_for_input in zip(data, real_output):
                update = self.learning_rate*(output_for_input - self.predict(input_given))
                self.weights[0] += update
                self.weights[1:] = update*input_given

    def predict(self, feature_list):
        """
        Predicts the output for a given input
        :param feature_list: the input
        :return: output, 1 or -1
        """
        result = self.weights[0] + self.weights[1:].dot(feature_list)
        output = 1 if result >= 0 else -1
        return output
