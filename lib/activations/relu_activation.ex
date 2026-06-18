defmodule ReluActivation do
  import Nx.Defn

  defn forward(value) do
    Nx.max(value, 0)
  end

  defn backward(dvalue, z) do
    dvalue * Nx.as_type(Nx.greater(z, 0), Nx.type(dvalue))
  end
end
