defmodule StepActivation do
  import Nx.Defn

  defn forward(value) do
    Nx.as_type(Nx.greater_equal(value, 0), Nx.type(value))
  end

  defn backward(_dvalue, _z) do
    0.0
  end
end
