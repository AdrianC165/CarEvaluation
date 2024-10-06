# CarEvaluation
We are exploring how suitable a car is to buy based on its features and pricing. 

## Project Overview
In this project, we’re analyzing the factors that influence car acceptability using a dataset from the UCI Machine Learning Repository. The dataset comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). The goal is to clean the data, build a neural network model, and predict car acceptability based on features such as price, comfort, and safety. We are also exploring activation functions (logistic, tanh, and relu) and evaluating model performance

## Dataset
We're using the "Car Evaluation Database" from the UCI ML Repository. You can download the dataset and place it in the appropiate folder (e.g., "CarEvaluation") before running the project:
- [Download the Dataset](https://archive.ics.uci.edu/dataset/19/car+evaluation)

## Required Libraries
You’ll need the following Python libraries to get everything running:
- `pandas` (for data handling)
- `numpy` (for numerical operations and array handling)
- `scikit-learn` (for machine learning tasks such as model training, splitting data, and standardization)
- `matplotlib` (for plotting)

You can install everything in one go with:
```bash
pip install pandas numpy scikit-learn matplotlib
```

## Project Structure
Ensure the following files and folders are set up properly in your project directory:
- `NeuralNet.py`: Contains the main code for data preprocessing, model building, training, and evaluation.
- `CarEvaluation/`: Folder containing the dataset files (e.g., `car.data`, `car.names`, etc.).
- `performance_log.csv`: This file will store the model performance results for different hyperparameters.


## Running the Code in VS Code

1. **Download the dataset** from the provided link and place it in the same directory as the main.py.

2. **Open VS Code** and navigate to the project directory.

3. **Set up a Python environment** (e.g., using a virtual environment) if you haven't already:
    ```bash
    python -m venv env
    source env/bin/activate  #On Windows, use `env\Scripts\activate`
    ```

4. **Install the required libraries** by running:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```

5. **Open the `NeuralNet.py` file in VS Code.**

6. **Run the Python file** using the built-in terminal. This will:
    - Load and clean the dataset.
    - Convert categorical variables into numerical format.
    - Standardize the features.
    - Split the data into training and test sets.
    - Train a neural network model using various hyperparameters.
    - Evaluate the model and generate plots.

7. **Check the output files** in the project directory:
    - `performance_log.csv`: Contains model performance metrics (training and test accuracies, errors) for different combinations of hyperparameters.
    - `model_accuracy_plot.png`: A plot showing training and test accuracy over epochs for each model configuration.


## Running the Code in Terminal
Follow the same steps as in VS Code. You can run `NeuralNet.py` file directly in the terminal with:
```bash
     python NeuralNet.py --dataFile ./CarEvaluation/car.data
```
    
## Notes
- Ensure that no hardcoded paths are used; the dataset should be placed in the `CarEvaluation` folder..
- The dataset file is not included in this repository but must be downloaded separately from the provided link.
