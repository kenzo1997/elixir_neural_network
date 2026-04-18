defmodule SoftmaxActivation do
  # Forward takes entire vector of z values
  def forward(z_values) do
    exp_values =
      Enum.map(z_values, fn z ->
        :math.exp(z)
      end)

    sum_exp = Enum.sum(exp_values)

    Enum.map(exp_values, fn e ->
      e / sum_exp
    end)
  end
end
