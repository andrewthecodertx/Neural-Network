# Go Neural Network

This project implements a feed-forward neural network in Go, designed to be
flexible and adaptable to various datasets. It currently supports training new
models and loading pre-trained models for prediction.

## Features

* **Feed-Forward Neural Network:** A basic implementation of a neural network
with input, hidden, and output layers.
* **Training:** Train the neural network using your own CSV data.
* **Model Persistence:** Save and load trained models to/from `model.json` files.
* **Prediction:** Use a loaded model to make predictions on new input data.
* **He Initialization:** Weights are initialized using He initialization for
ReLU activation functions.
* **ReLU Activation:** The hidden layer uses the Rectified Linear Unit (ReLU)
activation function.
* **Linear Output:** The output layer uses a linear activation function,
suitable for regression tasks.
* **Backpropagation:** Implements the backpropagation algorithm for training.

## Getting Started

To run this application, you need to have Go installed on your system.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/andrewthecodertx/go-neuralnetwork.git
    cd go-neuralnetwork
    ```

2. **Run the application:**

    ```bash
    go run .
    ```

## Training Data

The project includes a `data.csv` file, which is a sample dataset for training.
This data is based on the **Red Wine Quality dataset** from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).

**Dataset Description:**
The dataset contains physicochemical properties of red wines and their
corresponding quality ratings. It has 11 input features (e.g., fixed acidity,
volatile acidity, citric acid, alcohol) and 1 output feature
(quality, a score between 0 and 10).

## Usage

When you run the application, you will be prompted to either train a new model
or load an existing one:

```
Do you want to (t)rain a new model or (l)oad an existing model? (t/l):
```

### Training a New Model (`t`)

If you choose `t`, you will be asked for several parameters:

* **data file:** (default: `data.csv`) The path to your CSV training data.
* **inputs:** (default: 11) The number of input features in your dataset.
* **outputs:** (default: 1) The number of output features in your dataset.
* **hidden neurons:** (default: 16) The number of neurons in the hidden layer.
* **epochs:** (default: 200) The number of training iterations.
* **learning rate:** (default: 0.05) The learning rate for the backpropagation
algorithm.
* **error goal:** (default: 0.005) The target error to stop training early.

After training, the model will be saved to `model.json`.

### Loading an Existing Model (`l`)

If you choose `l`, the application will attempt to load `model.json`. Once
loaded, you can enter comma-separated input values for prediction:

```text
Model loaded from model.json
Enter input values for prediction (space-separated):
5.6,0.31,0.78,13.9,0.074,23.0,92.0,0.99677,3.39,0.48,10.5
Prediction for input: [predicted_value]
```

## Future Enhancements

* **Dynamic Network Architecture:** Allow the user to define the number of
hidden layers and neurons per layer.
* **Dockerization:** Provide a Docker image for easier deployment and use.
* **More Activation Functions:** Implement additional activation functions
(e.g., sigmoid, tanh).
* **Command-Line Arguments:** Allow all parameters to be passed via
command-line arguments instead of interactive prompts.

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to
open an issue or submit a pull request.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
