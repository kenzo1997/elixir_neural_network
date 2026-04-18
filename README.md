Neural Network in Elixir
A from-scratch implementation of a feedforward neural network in Elixir, built to train and classify handwritten digits from the MNIST dataset.
Overview
This project implements a complete neural network training pipeline without using any machine learning libraries. Everything from forward propagation to backpropagation and gradient descent is implemented manually using pure Elixir.
Key Features

Pure Elixir implementation - No ML frameworks, built from functional programming primitives
MNIST digit classification - Trains on the classic handwritten digit dataset
Multiple activation functions - ReLU, Softmax, Linear, and Step functions
Customizable architecture - Easily configure layer sizes and activation functions
Model persistence - Save and load trained models
Training visualization - Track loss and accuracy during training
ASCII digit preview - Visualize predictions in the terminal

Architecture
The default network architecture consists of:

Input layer: 784 neurons (28×28 pixel images flattened)
Hidden layer: 128 neurons with ReLU activation
Output layer: 10 neurons with Softmax activation (one per digit 0-9)

Project Structure
neural_network/
├── mix.exs                    # Project configuration
├── app.ex                     # Core neural network operations
├── neural_network.ex          # High-level training and prediction API
├── MnistLoader.ex            # MNIST dataset loader
├── relu_activation.ex         # ReLU activation function
├── softmax_activation.ex      # Softmax activation function
├── linear_activation.ex       # Linear activation function
└── step_function.ex           # Step activation function
Requirements

Elixir ~> 1.12
Erlang/OTP compatible with your Elixir version
MNIST dataset files (see Setup section)

Setup

Install dependencies:

bash   mix deps.get

Download MNIST dataset:
Create a dataset/ directory in your project root and download the following files:

train-images.idx3-ubyte - Training images
train-labels.idx1-ubyte - Training labels
t10k-images.idx3-ubyte - Test images (optional)
t10k-labels.idx1-ubyte - Test labels (optional)

Compile the project:

bash   mix compile
Usage
Training a Model
elixir# Start training with default parameters
NeuralNetwork.start()
This will:

Load 500 training samples from the MNIST dataset
Train for 50 epochs with learning rate 0.01
Display loss and accuracy after each epoch
Save the trained model to mnist_model.term

Training output example:
Epoch 1 | Loss: 245.32 | Accuracy: 12.4%
Epoch 2 | Loss: 198.76 | Accuracy: 34.2%
...
Epoch 50 | Loss: 12.45 | Accuracy: 94.8%
Model saved to mnist_model.term
Making Predictions
elixir# Load a trained model
layers = NeuralNetwork.load_model("mnist_model.term")

# Load test data
{_, _, _, test_images} = MnistLoader.load_images("dataset/t10k-images.idx3-ubyte")
{_, test_labels} = MnistLoader.load_labels("dataset/t10k-labels.idx1-ubyte")

# Predict a single image
prediction = NeuralNetwork.predict(layers, Enum.at(test_images, 0))
IO.puts("Predicted digit: #{prediction}")

# Predict with confidence score
result = NeuralNetwork.predict_with_confidence(layers, Enum.at(test_images, 0))
IO.puts("Digit: #{result.digit}, Confidence: #{result.confidence}")

# Test multiple samples
NeuralNetwork.test_samples(layers, test_images, test_labels, 10)
Visualizing Predictions
elixir# Load model and test data
layers = NeuralNetwork.load_model()
{_, _, _, images} = MnistLoader.load_images("dataset/t10k-images.idx3-ubyte")

# Print an image as ASCII art
NeuralNetwork.print_image(Enum.at(images, 0))
Customizing the Network
You can modify the network architecture by editing the NeuralNetwork.start/0 function:
elixirlayers = [
  %{
    weights: random_matrix(256, 784),  # 256 neurons in hidden layer
    bias: random_vector(256),
    activation: ReluActivation
  },
  %{
    weights: random_matrix(10, 256),
    bias: random_vector(10),
    activation: SoftmaxActivation
  }
]
Adjust hyperparameters:
elixirepochs = 100                # Number of training epochs
learning_rate = 0.005       # Learning rate for gradient descent
|> Enum.take(5000)          # Number of training samples
Implementation Details
Forward Propagation
For each layer:

Compute weighted sum: z = w·x + b
Apply activation function: a = activation(z)
Use output as input to next layer

Backpropagation

Compute loss derivative with respect to outputs
For each layer (in reverse):

Compute gradients for weights and biases
Propagate error to previous layer


Update weights and biases using gradient descent

Activation Functions

ReLU: f(x) = max(0, x) - Used in hidden layers
Softmax: f(x) = exp(x) / sum(exp(x)) - Used in output layer for classification
Linear: f(x) = x - Identity function
Step: f(x) = 1 if x > 0 else 0 - Binary threshold

Loss Functions

Mean Squared Error (MSE): loss = sum((predicted - actual)²)
Cross-Entropy Loss: loss = -sum(target * log(prediction)) - Better for classification

Limitations

Pure Elixir implementation is slower than optimized libraries like TensorFlow
No GPU acceleration
Limited to feedforward networks (no CNNs, RNNs, etc.)
Basic gradient descent (no Adam, RMSprop, etc.)

Future Improvements
Potential enhancements:

 Mini-batch gradient descent
 Advanced optimizers (Adam, RMSprop)
 Dropout regularization
 Learning rate scheduling
 Validation set evaluation
 More activation functions (Tanh, Leaky ReLU)
 Convolutional layers
 Better weight initialization (He, Xavier)

License
This project is provided as-is for educational purposes.
