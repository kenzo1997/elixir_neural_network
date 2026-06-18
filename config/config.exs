import Config

# Set EXLA as the default Nx backend for GPU acceleration.
# XLA detects CUDA at runtime and compiles kernels for the GPU automatically.
config :nx, :default_backend, EXLA.Backend
