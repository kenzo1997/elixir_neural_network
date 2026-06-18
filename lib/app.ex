defmodule App do
  import Nx.Defn

  @doc """
  Complete forward pass from raw inputs to softmax probabilities.

  Architecture: input -> ReLU hidden -> softmax output

  Shapes:
    inputs: {batch_size, 784}
    w1:     {784, 128}
    b1:     {1, 128}
    w2:     {128, 10}
    b2:     {1, 10}

  Returns: {batch_size, 10} — per-class probabilities.
  """
  defn forward(inputs, w1, b1, w2, b2) do
    z1 = Nx.dot(inputs, w1) + b1
    a1 = Nx.max(z1, 0)

    z2 = Nx.dot(a1, w2) + b2
    stable_softmax(z2)
  end

  @doc """
  Numerically stable softmax.
  Subtracts the per-row max before exponentiation to prevent overflow.
  """
  defn stable_softmax(logits) do
    shifted = logits - Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    exp = Nx.exp(shifted)
    exp / Nx.sum(exp, axes: [-1], keep_axes: true)
  end

  @doc """
  Cross-entropy loss averaged over the batch.

  predictions: {batch_size, 10}
  targets:     {batch_size, 10}  (one-hot)

  Returns: scalar
  """
  defn cross_entropy(predictions, targets) do
    eps = 1.0e-15
    clipped = Nx.max(predictions, eps)
    -Nx.mean(Nx.sum(targets * Nx.log(clipped), axes: [-1]))
  end

  @doc """
  Mean Squared Error loss averaged over the batch.
  """
  defn mse(predictions, targets) do
    Nx.mean(Nx.pow(predictions - targets, 2))
  end

  @doc """
  Manual backward pass — computes gradients for all parameters.

  Derivation (combined softmax + cross-entropy):
    dL/dz2 = softmax - targets            (per-sample, {batch, 10})
    dL/dw2 = a1^T @ dL/dz2 / batch_size
    dL/db2 = mean(dL/dz2, axis=0)

    dL/da1 = dL/dz2 @ w2^T
    dL/dz1 = dL/da1 * (z1 > 0)           (ReLU backward)
    dL/dw1 = inputs^T @ dL/dz1 / batch_size
    dL/db1 = mean(dL/dz1, axis=0)

  Returns: {dW1, dB1, dW2, dB2} — same shapes as the parameters.
  """
  defn backward(inputs, targets, w1, b1, w2, b2) do
    # Forward pass (recomputed for the gradient tape)
    z1 = Nx.dot(inputs, w1) + b1
    a1 = Nx.max(z1, 0)

    z2 = Nx.dot(a1, w2) + b2
    softmax = stable_softmax(z2)

    # --- Output layer gradients ---
    dZ2 = softmax - targets

    dW2 = Nx.dot(Nx.transpose(a1), dZ2)
    dB2 = Nx.mean(dZ2, axes: [0], keep_axes: true)

    # --- Hidden layer gradients ---
    dA1 = Nx.dot(dZ2, Nx.transpose(w2))
    relu_mask = Nx.as_type(Nx.greater(z1, 0), Nx.type(dA1))
    dZ1 = dA1 * relu_mask

    dW1 = Nx.dot(Nx.transpose(inputs), dZ1)
    dB1 = Nx.mean(dZ1, axes: [0], keep_axes: true)

    # Normalize by batch size
    batch_size = Nx.axis_size(inputs, 0) |> Nx.as_type(Nx.type(dW1))
    {dW1 / batch_size, dB1, dW2 / batch_size, dB2}
  end

  @doc """
  Gradient descent update for one layer's weights and bias.
  """
  def update_layer(weights, bias, dw, db, learning_rate) do
    {
      weights - learning_rate * dw,
      bias - learning_rate * db
    }
  end
end
