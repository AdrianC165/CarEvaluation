#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
import csv                                              # Used to append (add on) to file that tracks performance
from sklearn.neural_network import MLPClassifier        # Used to create neural Network
import matplotlib.pyplot as plt                         # Used to plot performance history
from sklearn.model_selection import train_test_split


class NeuralNet:
    def __init__(self, dataFile, header=True):
        try:
            self.raw_input = pd.read_csv(dataFile, header=None)
        except:
            print(f"File {dataFile} not found. Please ensure the file path is correct.")
       

    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        # Add correct column headers manually since 'car.data' does not have headers
        headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'CAR']
        self.raw_input.columns = headers  # Assign headers to the DataFrame
        
        # Convert categorical variables to numerical variables
        self.processed_data = self.raw_input
        self.processed_data['buying'] = self.processed_data['buying'].map({'vhigh': 3, 'high': 2, 'med': 1, 'low': 0})
        self.processed_data['maint'] = self.processed_data['maint'].map({'vhigh': 3, 'high': 2, 'med': 1, 'low': 0})
        self.processed_data['doors'] = self.processed_data['doors'].map({'2': 0, '3': 1, '4': 2, '5more': 3})
        self.processed_data['persons'] = self.processed_data['persons'].map({'2': 0, '4': 1, 'more': 2})
        self.processed_data['lug_boot'] = self.processed_data['lug_boot'].map({'small': 0, 'med': 1, 'big': 2})
        self.processed_data['safety'] = self.processed_data['safety'].map({'low': 0, 'med': 1, 'high': 2})
        self.processed_data['CAR'] = self.processed_data['CAR'].map({'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3})

        # Check for NaN values (only prints them out)
        print(self.processed_data.isnull().sum())
        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4)

        # hyperparameters used for model evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Tracks performance
        history = []
        file_name = "performance_log.csv"

        # Helper function to log results
        def log_results(filename, hyperparams, train_acc, train_err, test_acc, test_err):
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                # Round the accuracy and error values to 3 decimal places
                writer.writerow([hyperparams, round(train_acc, 3), round(train_err, 3), round(test_acc, 3), round(test_err, 3)])
        
        for activation in activations:
            
            for lr in learning_rate:
                for epochs in max_iterations:
                    for layers in num_hidden_layers:
                        # Create MLP model
                        model = MLPClassifier(activation=activation, 
                                              learning_rate_init=lr, 
                                              max_iter=epochs, 
                                              hidden_layer_sizes=(layers,))
                        
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Get accuracy and error/loss
                        train_acc = model.score(X_train, y_train)
                        test_acc = model.score(X_test, y_test)
                        train_err = 1 - train_acc
                        test_err = 1 - test_acc
                        
                        # Log the results
                        hyperparams = f"Activation={activation}, LR={lr}, Epochs={epochs}, Layers={layers}"
                        log_results(file_name, hyperparams, train_acc, train_err, test_acc, test_err)
                        
                        # Track history for plotting
                        history.append((epochs, train_acc, test_acc))

        # Plot model history (accuracy vs epochs)
        epochs_list, train_acc_list, test_acc_list = zip(*history)
        plt.plot(epochs_list, train_acc_list, label="Train Accuracy")
        plt.plot(epochs_list, test_acc_list, label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        return 0




if __name__ == "__main__":
    neural_network = NeuralNet("./CarEvaluation/car.data") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
