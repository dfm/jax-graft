# jax-graft: run JAX on your Apple GPU

A custom backend for JAX adding support for Apple GPUs.

Under the hood it treats Metal as an XLA:GPU platform (called `"mtl"`),
next to CUDA and ROCm. XLA's own GPU compiler fuses your program and
generates the kernels, which the plugin translates to Metal Shading
Language.

This project isn't affiliated with or endorsed by Apple or Google.

## Status

`jax-graft` has wide coverage of the JAX test suite (with the notable
exception of float64 support), but you should expect some sharp edges.
The performance is broadly comparable to MLX and PyTorch's MPS backend
on the workloads we've tried. [Please report issues](CONTRIBUTING.md).

## Install

```
pip install jax-graft
```

## Quick start

```python
import jax
import jax.numpy as jnp

x = jnp.arange(4.0)
y = jax.jit(lambda v: v * 2)(x)
print(y, y.devices())  # [0. 2. 4. 6.] {MtlDevice(id=0)}
```

Just installing `jax-graft` is enough to set the default backend to
`"mtl"` on Macs with Apple GPUs. To make CPU the default, set the
`JAX_PLATFORMS=cpu,mtl` environment variable, and place work on the
GPU explicitly using `jax.device_put(x, jax.devices("mtl")[0])`.

Some more complete worked examples can be found in the
[`examples`](examples) directory, and the [`docs`](docs) directory
has a lot more information.

## Contributing

Issues and pull requests are welcome. [`CONTRIBUTING.md`](CONTRIBUTING.md)
says what makes a useful report, and
[`docs/development.md`](docs/development.md) covers building and testing.

## License

Apache-2.0 ([`LICENSE`](LICENSE)). The plugin includes third-party code:
some kernels ported from [MLX](https://github.com/ml-explore/mlx) (MIT),
Apple's metal-cpp headers (Apache-2.0), and a statically linked XLA. Their
notices are in [`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES), which ships in
both wheels.
