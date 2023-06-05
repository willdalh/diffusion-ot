import copy


# EMA code inspired by https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
class EMA:
    """
    Exponential moving average
    """
    def __init__(self, beta, model):
        """
        Args:
            beta (float): decay rate
            model (torch.nn.Module): model to apply EMA to
        """
        self.beta = beta
        self.step = 0
        self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
        self.step_start = 2000

    def update_model_average(self, model):
        """
        Update EMA model with exponential moving average of input model

        Args:
            model (torch.nn.Module): model to use as source of parameters
        """
        for current_param, ema_param in zip(model.parameters(), self.ema_model.parameters()):
            old_weight, new_weight = ema_param.data, current_param.data
            ema_param.data = self.update_average(old_weight, new_weight)


    def update_average(self, old, new):
        """
        Update exponential moving average on specific parameter tensor

        Args:
            old (torch.Tensor): current parameter tensor
            new (torch.Tensor): new parameter tensor

        Returns:
            torch.Tensor, EMA updated parameter tensor
        """
        return old * self.beta + new * (1 - self.beta)

    def step_ema(self, model):
        """
        Perform an EMA step
        EMA is applied after step_start steps

        Args:
            model: torch.nn.Module, model to use as source of parameters
        """
        self.step += 1
        if self.step < self.step_start:
            self.reset_parameters(model)
            return

        self.update_model_average(model)

    def reset_parameters(self, model):
        """
        Reset EMA parameters to current model parameters
        """
        self.ema_model.load_state_dict(model.state_dict())