defmodule NeuralNetwork do
  alias App
  alias ReluActivation
  alias LinearActivation
  alias SoftmaxActivation

  def forward_network(layers, inputs) do
    Enum.reduce(layers, {inputs, []}, fn layer, {curr_inputs, caches} ->
      # Compute Z for each neuron
      z_outputs =
        Enum.zip(layer.weights, layer.bias)
        |> Enum.map(fn {w, b} ->
          z =
            Enum.zip(curr_inputs, w)
            |> Enum.map(fn {i, w} -> i * w end)
            |> Enum.sum()

          z + b
        end)

      # Apply activation
      activations =
        if layer.activation == SoftmaxActivation do
          SoftmaxActivation.forward(z_outputs)
        else
          Enum.map(z_outputs, fn z -> layer.activation.forward(z) end)
        end

      # Keep cache for backprop
      outputs = Enum.zip(z_outputs, activations)
      cache = %{
        outputs: outputs,
        inputs: curr_inputs
      }

      {activations, caches ++ [cache]}
    end)
  end

  def backward_network(layers, caches, outputs, targets) do
    # Derivative of loss wrt final activations
    # dvalues =
    #   Enum.zip(outputs, targets)
    #   |> Enum.map(fn {a, t} -> App.loss_derivative(a, t) end)

    dvalues =
      Enum.zip(outputs, targets)
      |> Enum.map(fn {y, t} -> y - t end)

    {gradients, _} =
      layers
      |> Enum.with_index()
      |> Enum.reverse()
      |> Enum.reduce({[], dvalues}, fn {layer, i}, {grads_acc, dvals} ->
        cache = Enum.at(caches, i)

        gradients =
          if layer.activation == SoftmaxActivation do
            App.backward_layer(dvals, cache.outputs, cache.inputs, LinearActivation)
          else
            App.backward_layer(dvals, cache.outputs, cache.inputs, layer.activation)
          end

        dz = Enum.map(gradients, fn {_dws, db} -> db end)
        new_dvalues = App.propagate_dvalues(dz, layer.weights)

        {[gradients | grads_acc], new_dvalues}
      end)

    gradients
  end

  def update_network(layers, gradients, learning_rate) do
    Enum.zip(layers, gradients)
    |> Enum.map(fn {layer, grads} ->
      updated = App.update_layer(layer.weights, layer.bias, grads, learning_rate)

      %{
        layer |
        weights: Enum.map(updated, fn {w, _} -> w end),
        bias: Enum.map(updated, fn {_, b} -> b end)
      }
    end)
  end

  def random_matrix(rows, cols) do
    for _ <- 1..rows do
      for _ <- 1..cols do
        (:rand.uniform() - 0.5) * 0.01
      end
    end
  end
  
  def random_vector(size) do
    for _ <- 1..size do
      0.0
    end
  end

  def load_model(path \\ "mnist_model.term") do
    path
    |> File.read!()
    |> :erlang.binary_to_term()
  end

  def predict(layers, inputs) do
    {outputs, _} = forward_network(layers, inputs)
  
    outputs
    |> Enum.with_index()
    |> Enum.max_by(fn {v, _i} -> v end)
    |> elem(1)
  end

  def predict_with_confidence(layers, inputs) do
    {outputs, _} = forward_network(layers, inputs)
  
    {prob, index} =
      outputs
      |> Enum.with_index()
      |> Enum.max_by(fn {v, _i} -> v end)
  
    %{digit: index, confidence: prob}
  end

  def start do
    epochs = 50
    learning_rate = 0.01

    {_, _, _, images} =
      MnistLoader.load_images("dataset/train-images.idx3-ubyte")
  
    {_, labels} =
      MnistLoader.load_labels("dataset/train-labels.idx1-ubyte")
  
    dataset =
     Enum.zip(
       images,
       Enum.map(labels, &MnistLoader.one_hot/1)
     )
     |> Enum.take(500)   # start small

    layers = [
      %{
        weights: random_matrix(128, 784),
        bias: random_vector(128),
        activation: ReluActivation
      },
      %{
        weights: random_matrix(10, 128),
        bias: random_vector(10),
        activation: SoftmaxActivation
      }
    ]


    final_layers = Enum.reduce(1..epochs, layers, fn epoch, layers ->
      {updated_layers, total_loss, total_correct} =
        Enum.reduce(dataset, {layers, 0, 0}, fn {inputs, targets}, {curr_layers, loss_acc, correct_acc} ->
          {outputs, caches} = forward_network(curr_layers, inputs)
          loss = App.loss(outputs, targets)
  
          # --- Accuracy calculation ---
          predicted =
            outputs
            |> Enum.with_index()
            |> Enum.max_by(fn {v, _i} -> v end)
            |> elem(1)
  
          actual =
            targets
            |> Enum.with_index()
            |> Enum.max_by(fn {v, _i} -> v end)
            |> elem(1)
  
          correct = if predicted == actual, do: 1, else: 0
          # --- End accuracy calculation ---
  
          gradients = backward_network(curr_layers, caches, outputs, targets)
          new_layers = update_network(curr_layers, gradients, learning_rate)
  
          {new_layers, loss_acc + loss, correct_acc + correct}
        end)
  
      accuracy_percent = total_correct / length(dataset) * 100
      IO.puts("Epoch #{epoch} | Loss: #{total_loss} | Accuracy: #{accuracy_percent}%")
      updated_layers
    end)

    IO.inspect(final_layers, label: "Final trained layers")

    File.write!(
      "mnist_model.term",
      :erlang.term_to_binary(final_layers)
    )

    IO.puts("Model saved to mnist_model.term")
  end

  def test_samples(layers, images, labels, count \\ 5) do
    results =
      images
      |> Enum.zip(labels)
      |> Enum.take(count)
      |> Enum.with_index(1)
      |> Enum.map(fn {{img, label}, index} ->
        prediction = predict(layers, img)
        correct? = prediction == label
  
        IO.puts("\nSample #{index}")
        IO.puts("Predicted: #{prediction} | Actual: #{label} | Correct: #{correct?}")
  
        #print_image(img)
  
        correct?
      end)
  
    total = length(results)
    correct = Enum.count(results, & &1)
    accuracy = correct / total * 100
  
    IO.puts("\n==============================")
    IO.puts("#{correct}/#{total} correct. #{Float.round(accuracy, 2)}% accuracy")
    IO.puts("==============================")
  end

  def print_image(image) do
    image
    |> Enum.chunk_every(28)
    |> Enum.each(fn row ->
      row
      |> Enum.map(fn pixel ->
        cond do
          pixel > 0.7 -> "█"
          pixel > 0.3 -> "▓"
          pixel > 0.1 -> "▒"
          pixel > 0.05 -> "."
          true -> " "
        end
      end)
      |> Enum.join("")
      |> IO.puts()
    end)
  end
end
