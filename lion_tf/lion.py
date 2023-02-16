import tensorflow as tf
from keras.optimizers.optimizer_experimental import optimizer


def lerp(start, end, weight):
    return start + weight * (end - start)


def sparse_lerp(start, end, weight):
    return start + weight * -(start - end)


class Lion(optimizer.Optimizer):
    r"""Optimizer that implements the Lion algorithm.
    Lion was published in the paper "Symbolic Discovery of Optimization Algorithms"
    which is available at https://arxiv.org/abs/2302.06675
    Args:
      learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to 1e-4.
      beta_1: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. Factor
         used to interpolate the current gradient and the momentum. Defaults to 0.9.
      beta_2: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the momentum. Defaults to 0.99.
    Notes:
    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    """

    def __init__(
        self,
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.99,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        jit_compile=True,
        name="Lion",
        **kwargs
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def build(self, var_list):
        """Initialize optimizer variables.
          var_list: list of model variables to build Lion variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._emas = []
        for var in var_list:
            self._emas.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="ema"
                )
            )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        beta_1 = tf.constant(self.beta_1, shape=(1,))
        beta_2 = tf.constant(self.beta_2, shape=(1,))

        var_key = self._var_key(variable)
        ema = self._emas[self._index_dict[var_key]]

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            lerp_fn = sparse_lerp
        else:
            # Dense gradients.
            lerp_fn = lerp

        update = lerp_fn(ema, gradient, 1 - beta_1)
        update = tf.sign(update)
        variable.assign_sub(update * lr)

        ema.assign(lerp_fn(ema, gradient, 1 - beta_2))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
            }
        )
        return config