defmodule MnistLoader do
  def load_images(path) do
    {:ok, binary} = File.read(path)

    <<_magic::32, num_images::32, rows::32, cols::32, rest::binary>> = binary

    image_size = rows * cols

    images =
      rest
      |> :binary.bin_to_list()
      |> Enum.chunk_every(image_size)
      |> Enum.take(num_images)
      |> Enum.map(fn pixels ->
        Enum.map(pixels, fn p -> p / 255.0 end)
      end)

    {num_images, rows, cols, Nx.tensor(images, type: {:f, 32})}
  end

  def load_labels(path) do
    {:ok, binary} = File.read(path)

    <<_magic::32, num_labels::32, rest::binary>> = binary

    labels =
      rest
      |> :binary.bin_to_list()
      |> Enum.take(num_labels)

    {num_labels, Nx.tensor(labels, type: {:s, 64})}
  end

  def one_hot(label) do
    for i <- 0..9 do
      if i == label, do: 1.0, else: 0.0
    end
  end

  def one_hot_tensor(labels) do
    num_labels = Nx.axis_size(labels, 0)
    cols = Nx.tensor(0..9, type: {:s, 64})
    Nx.as_type(
      Nx.equal(Nx.reshape(labels, {num_labels, 1}), cols),
      {:f, 32}
    )
  end
end
