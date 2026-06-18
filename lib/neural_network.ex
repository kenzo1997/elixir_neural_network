defmodule NeuralNetwork do
  alias App

  @input_size  784
  @hidden_size 128
  @output_size 10

  # ── Initialization ──────────────────────────────────────────────

  @doc """
  Creates a randomly-initialized 2-layer network.

  Returns a map:
    %{w1: {784,128}, b1: {1,128}, w2: {128,10}, b2: {1,10}}
  """
  def init_layers(opts \\ []) do
    input_size  = opts[:input_size]  || @input_size
    hidden_size = opts[:hidden_size] || @hidden_size
    output_size = opts[:output_size] || @output_size

    %{
      w1: Nx.random_uniform({input_size, hidden_size}, -0.005, 0.005, type: {:f, 32}),
      b1: Nx.broadcast(0.0, {1, hidden_size}),
      w2: Nx.random_uniform({hidden_size, output_size}, -0.005, 0.005, type: {:f, 32}),
      b2: Nx.broadcast(0.0, {1, output_size})
    }
  end

  # ── Forward pass (predict path) ─────────────────────────────────

  @doc """
  Runs forward propagation across all layers.
  Input can be a single sample {784} or a batch {N, 784}.
  Returns the softmax probability vector(s).
  """
  def forward_network(layers, inputs) do
    # Ensure batch dimension
    inputs = if Nx.rank(inputs) == 1, do: Nx.new_axis(inputs, 0), else: inputs
    App.forward(inputs, layers.w1, layers.b1, layers.w2, layers.b2)
  end

  # ── Prediction ──────────────────────────────────────────────────

  @doc """
  Predict the digit (0-9) for a single input tensor of shape {784}.
  """
  def predict(layers, inputs) do
    inputs = Nx.new_axis(inputs, 0)
    outputs = forward_network(layers, inputs)
    outputs
    |> Nx.argmax(axis: -1)
    |> Nx.squeeze(axes: [0])
    |> Nx.to_number()
  end

  @doc """
  Predict with confidence score.

  Returns: %{digit: integer, confidence: float}
  """
  def predict_with_confidence(layers, inputs) do
    inputs = Nx.new_axis(inputs, 0)
    outputs = forward_network(layers, inputs) |> Nx.squeeze(axes: [0])
    {prob, idx} =
      outputs
      |> Nx.to_flat_list()
      |> Enum.with_index()
      |> Enum.max_by(fn {v, _i} -> v end)

    %{digit: idx, confidence: prob}
  end

  # ── Training ────────────────────────────────────────────────────

  @doc """
  Train the network on MNIST data.

  Options:
    :epochs       — training epochs       (default 30)
    :batch_size   — mini-batch size       (default 64)
    :learning_rate — SGD learning rate    (default 0.01)
    :num_samples  — images to train on    (default 5000)
    :shuffle      — shuffle each epoch    (default true)
  """
  def start(opts \\ []) do
    epochs        = opts[:epochs]        || 30
    batch_size    = opts[:batch_size]    || 64
    learning_rate = opts[:learning_rate] || 0.01
    num_samples   = opts[:num_samples]   || 5000
    do_shuffle    = Keyword.get(opts, :shuffle, true)

    IO.puts("Loading MNIST data...")
    {num_total, _, _, images} = MnistLoader.load_images("dataset/train-images.idx3-ubyte")
    {_, labels}               = MnistLoader.load_labels("dataset/train-labels.idx1-ubyte")

    IO.puts("Total available: #{num_total} samples")
    actual_samples = min(num_samples, num_total)
    IO.puts("Using: #{actual_samples} samples")

    # Take a subset and convert labels to one-hot
    images = Nx.slice(images,  [0, 0], [actual_samples, 784])
    labels = MnistLoader.one_hot_tensor(
               Nx.slice(labels, [0], [actual_samples]))
          |> Nx.as_type({:f, 32})

    # Initialize weights
    layers = init_layers()
    IO.puts("Network: #{@input_size} -> #{@hidden_size} -> #{@output_size}")
    IO.puts("Batch size: #{batch_size}, Epochs: #{epochs}, LR: #{learning_rate}\n")

    num_batches = div(actual_samples, batch_size)

    final_layers =
      Enum.reduce(1..epochs, layers, fn epoch, layers ->
        # Shuffle dataset
        {epoch_images, epoch_labels} =
          if do_shuffle do
            indices = Enum.shuffle(0..(actual_samples - 1))
            idx_tensor = Nx.tensor(indices, type: {:s, 64})
            {Nx.take(images, idx_tensor, axis: 0), Nx.take(labels, idx_tensor, axis: 0)}
          else
            {images, labels}
          end

        # Train on mini-batches
        {layers, epoch_loss, epoch_correct} =
          Enum.reduce(0..(num_batches - 1), {layers, 0.0, 0}, fn b, {layers, loss_acc, correct_acc} ->
            start_idx = b * batch_size

            batch_x = Nx.slice(epoch_images, [start_idx, 0], [batch_size, 784])
            batch_y = Nx.slice(epoch_labels, [start_idx, 0], [batch_size, 10])

            # Forward + loss
            outputs = App.forward(batch_x, layers.w1, layers.b1, layers.w2, layers.b2)
            loss     = App.cross_entropy(outputs, batch_y) |> Nx.to_number()

            # Backward
            {dw1, db1, dw2, db2} =
              App.backward(batch_x, batch_y, layers.w1, layers.b1, layers.w2, layers.b2)

            # Gradient descent update
            layers = %{
              w1: layers.w1 - learning_rate * dw1,
              b1: layers.b1 - learning_rate * db1,
              w2: layers.w2 - learning_rate * dw2,
              b2: layers.b2 - learning_rate * db2
            }

            # Accuracy for this batch
            preds  = Nx.argmax(outputs, axis: -1)
            actual = Nx.argmax(batch_y, axis: -1)
            correct = Nx.sum(Nx.equal(preds, actual)) |> Nx.to_number()

            {layers, loss_acc + loss, correct_acc + correct}
          end)

        total_samples = num_batches * batch_size
        avg_loss = epoch_loss / max(num_batches, 1)
        accuracy = epoch_correct / max(total_samples, 1) * 100
        IO.puts("Epoch #{epoch} | Loss: #{Float.round(avg_loss, 4)} | Accuracy: #{Float.round(accuracy, 2)}%")
        layers
      end)

    # Save model
    save_model(final_layers)
    IO.puts("\nModel saved to mnist_model.term")
    final_layers
  end

  # ── Testing ─────────────────────────────────────────────────────

  @doc """
  Run predictions on a set of test images and print results.
  """
  def test_samples(layers, images, labels, count \\ 10) do
    images = Nx.slice(images,  [0, 0], [count, 784])
    labels = Nx.slice(labels, [0],   [count])

    outputs = forward_network(layers, images)
    preds = Nx.argmax(outputs, axis: -1) |> Nx.to_flat_list()

    actuals = Nx.to_flat_list(labels)

    results =
      Enum.zip(preds |> Enum.with_index(1), actuals)
      |> Enum.map(fn {{pred, idx}, actual} ->
        correct = pred == actual

        IO.puts("Sample #{idx}: Predicted #{pred} | Actual #{actual} | #{if correct, do: "✓", else: "✗"}")
        correct
      end)

    total   = length(results)
    correct = Enum.count(results, & &1)
    pct     = Float.round(correct / max(total, 1) * 100, 2)

    IO.puts("\n#{correct}/#{total} correct — #{pct}% accuracy")
  end

  # ── Visualization ───────────────────────────────────────────────

  @doc """
  Render a 28×28 image as ASCII art in the terminal.
  Input: {784} tensor or flat list of 784 pixel values in [0, 1].
  """
  def print_image(image) do
    pixels =
      if is_struct(image, Nx.Tensor) do
        Nx.to_flat_list(image)
      else
        image
      end

    pixels
    |> Enum.chunk_every(28)
    |> Enum.each(fn row ->
      row
      |> Enum.map(fn p ->
        cond do
          p > 0.7 -> "█"
          p > 0.3 -> "▓"
          p > 0.1 -> "▒"
          p > 0.05 -> "."
          true -> " "
        end
      end)
      |> Enum.join("")
      |> IO.puts()
    end)
  end

  # ── Persistence ─────────────────────────────────────────────────

  @doc """
  Save model layers to a binary file.
  """
  def save_model(layers, path \\ "mnist_model.term") do
    binary = :erlang.term_to_binary(layers)
    File.write!(path, binary)
    IO.puts("Model saved to #{path}")
  end

  @doc """
  Load model layers from a binary file.
  """
  def load_model(path \\ "mnist_model.term") do
    path
    |> File.read!()
    |> :erlang.binary_to_term()
  end
end
