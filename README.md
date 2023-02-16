# lion-tf
A TensorFlow implementation of the Lion optimizer from [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675). Partially copied from the [lucidrains PyTorch implementation](https://github.com/lucidrains/lion-pytorch).

The maths seem right and it successfully trained a couple of models for me, but that doesn't mean I haven't forgotten something stupid, or that there isn't room for optimization!

In general, the code trusts in :pray:XLA:pray: to efficiently reuse buffers and save memory rather than manually doing all the ops in-place like the PyTorch version does. Note that the optimizer will be compiled with XLA even if you don't use `jit_compile` for the rest of your model!

## Installation
`pip install git+https://github.com/Rocketknight1/lion-tf.git`

## Usage
```python
from lion_tf import Lion

model.compile(Lion(1e-5))
```

## Tips

Lion likes much lower learning rates than Adam - I'd suggest a factor of 10 lower as a good starting point. When
fine-tuning pre-trained models, learning rates are already quite low, which means the optimal LR for Lion can be
*very* low. I found 1e-5 or less worked well for fine-tuning BERT!
