defmodule SoftmaxActivation do
  import Nx.Defn

  defn forward(z_values) do
    shifted = z_values - Nx.reduce_max(z_values, axes: [-1], keep_axes: true)
    exp = Nx.exp(shifted)
    exp / Nx.sum(exp, axes: [-1], keep_axes: true)
  end

  # Softmax backward is combined with cross-entropy loss gradient in practice.
  # Standalone: Jacobian matrix — not commonly used directly.
end
