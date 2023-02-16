# lion-tf
A TensorFlow implementation of the Lion optimizer from [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675). Partially copied from the [lucidrains PyTorch implementation](https://github.com/lucidrains/lion-pytorch).

The maths seem right and it successfully trained a couple of models for me, but that doesn't mean I haven't forgotten something stupid, or that there isn't room for optimization!

In general, the code trusts in :pray:XLA:pray: to efficiently reuse buffers and save memory rather than manually doing all the ops in-place like the PyTorch version does.

## Installation
`pip install git+https://github.com/Rocketknight1/lion-tf.git`

## Usage
```python
from lion_tf import Lion

model.compile(Lion(1e-5))
```
