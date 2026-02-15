defmodule StepActivation do
  def forward(value) do
    if value >= 0, do: 1, else: 0
  end

  def backward(_dvalue, _z) do
    0
  end
end
