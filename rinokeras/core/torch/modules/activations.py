import torch

class GatedTanh(torch.nn.Module):
    def __init__(self, input_units: int, output_units: int):
        """
        Gated TanH activation layer from https://arxiv.org/pdf/1612.08083.pdf
        
        Arguments:
            input_units {int} -- The number of input units to the module
            output_units {int} -- The numer of output units from the module
        """
        super(GatedTanh, self).__init__()
        self.input_units = input_units
        self.output_units = output_units

        # The linear input and output gates
        self.linear_forward = torch.nn.Linear(self.input_units, self.output_units, bias=True)
        self.gate = torch.nn.Linear(self.input_units, self.output_units, bias=True)

    def forward(self, *inputs):
        forward_out = torch.nn.functional.tanh(self.linear_forward(*inputs))
        gate_out = torch.nn.functional.sigmoid(self.gate(*inputs))
        return forward_out * gate_out