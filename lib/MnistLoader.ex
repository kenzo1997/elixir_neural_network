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

    {num_images, rows, cols, images}
  end

  def load_labels(path) do
    {:ok, binary} = File.read(path)
  
    <<_magic::32, num_labels::32, rest::binary>> = binary
  
    labels =
      rest
      |> :binary.bin_to_list()
      |> Enum.take(num_labels)
  
    {num_labels, labels}
  end

  def one_hot(label) do
    for i <- 0..9 do
      if i == label, do: 1.0, else: 0.0
    end
  end
end
