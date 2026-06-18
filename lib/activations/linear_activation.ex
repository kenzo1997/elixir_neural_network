defmodule LinearActivation do
  import Nx.Defn

  defn forward(value) do
    value
  end

  defn backward(dvalue, _z) do
    dvalue
  end
end
