defmodule App do
  defp inner(inputs, weights, bias, func) do
    z =
      Enum.zip(inputs, weights)
      |> Enum.map(fn {i, w} -> i * w end)
      |> Enum.sum()
      + bias    # add bias after sum
  
    a = func.(z)  # apply activation
    {z, a}        # return both
  end

  def forward_pass(inputs, weights, bias, func) do
    Enum.zip(weights, bias)
    |> Enum.map(fn {w, b} ->
      inner(inputs, w, b, func)
    end)
  end

  def forward_network(inputs, weights_list, bias_list, activations) do
    Enum.zip(weights_list, Enum.zip(bias_list, activations))
    |> Enum.reduce(inputs, fn {layer_weights, {layer_bias, activation}}, acc ->
      forward_pass(acc, layer_weights, layer_bias, activation) end) 
    end
  
  def backward_layer(dvalues, layer_outputs, inputs, activation) do
    Enum.zip(dvalues, layer_outputs)
    |> Enum.map(fn {dvalue, {z, _a}} ->
      backward_neuron(dvalue, z, inputs, activation)
    end)
  end
 
  def backward_neuron(dvalue, z, inputs, activation) do
    dL_dz = activation.backward(dvalue, z)

    dL_dw = Enum.map(inputs, fn x -> x * dL_dz end)
    dL_db = dL_dz

    {dL_dw, dL_db}
  end

  def loss_derivative(predicted, target) do
    2 * (predicted - target)
  end

  def update_layer(weights, biases, gradients, learning_rate) do
    Enum.zip(weights, Enum.zip(biases, gradients))
    |> Enum.map(fn {neuron_weights, {bias, {dws, db}}} ->
      new_weights =
        Enum.zip(neuron_weights, dws)
        |> Enum.map(fn {w, dw} ->
          w - learning_rate * dw
        end)

      new_bias = bias - learning_rate * db

      {new_weights, new_bias}
    end)
  end

  def propagate_dvalues(dz, weights) do
    0..(length(List.first(weights)) - 1)
    |> Enum.map(fn j ->
      Enum.zip(dz, weights)
      |> Enum.reduce(0, fn {dz_k, w_k}, acc ->
        acc + dz_k * Enum.at(w_k, j)
      end)
    end)
  end

  def loss(predicted, targets) do
    Enum.zip(predicted, targets)
    |> Enum.map(fn {p, t} -> (p - t) ** 2 end)
    |> Enum.sum()
  end

  def cross_entropy_loss(predictions, targets) do
    Enum.zip(predictions, targets)
    |> Enum.map(fn {p, t} ->
      # avoid log(0)
      p = max(p, 1.0e-15)
      -t * :math.log(p)
    end)
    |> Enum.sum()
  end
end


#-------------------------------------------------------------------------------
