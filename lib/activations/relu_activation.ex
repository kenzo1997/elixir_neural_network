defmodule ReluActivation do
  def forward(value) do
    if value > 0, do: value, else: 0
  end

  def backward(dvalue, z) do
    if z > 0, do: dvalue, else: 0
  end
end
