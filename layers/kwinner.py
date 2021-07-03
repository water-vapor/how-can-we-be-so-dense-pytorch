import torch

from ops import flat_kwinner_mask


class KWinner(torch.nn.Module):
    def __init__(
            self, k, boost=True, alpha=1 / 1000, beta=1.0, k_inference_factor=1.0
    ):
        """

        Args:
            k: active neurons to keep
            boost: whether to use boosted version of k-winner
            alpha: 1/T, where T=1000, ref: https://doi.org/10.3389/fncom.2017.00111
            beta: boost factor
            k_inference_factor: multiply k by this factor in inference
            **kwargs:
        """
        super().__init__()
        self.k = k
        self.boost = boost
        self.alpha = alpha
        self.beta = beta
        self.k_inference_factor = k_inference_factor
        self.initialized = False
        self.filters = None
        self.units = None
        self.target_duty_cycle = None
        self.duty_cycle = None

    def _initialize(self, input_shape):
        if not self.units:
            self.filters = input_shape[-1]
            self.units = torch.prod(torch.tensor(input_shape[1:]))
        self.target_duty_cycle = self.k / self.units
        self.target_duty_cycle = self.target_duty_cycle.type(torch.float32)
        self.duty_cycle = torch.zeros(self.filters)
        self.initialized = True

    def forward(self, inputs):
        if not self.initialized:
            self._initialize(inputs.shape)

        if not self.training:
            k = int(self.k * self.k_inference_factor)
        else:
            k = self.k

        if self.units <= k:
            return torch.nn.functional.relu(inputs)
        if self.boost:
            boost_term = torch.exp(self.beta * (self.target_duty_cycle - self.duty_cycle))
            boosted_inputs = inputs * boost_term
            boolean_mask = flat_kwinner_mask(boosted_inputs, k)
            if self.training:
                current_duty = torch.mean(boolean_mask.type(torch.float32), tuple(range(0, boolean_mask.ndim - 1)))
                self.duty_cycle = (1 - self.alpha) * self.duty_cycle + self.alpha * current_duty

        else:
            boolean_mask = flat_kwinner_mask(inputs, k)

        return torch.where(boolean_mask, inputs, torch.tensor(0.))
