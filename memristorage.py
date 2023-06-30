import torch.nn as nn
import math

class Memristorage:
    def __init__(self, model: nn.Module, crossbar_size: int=512) -> None:
        self.crossbar_size = crossbar_size
        self.width_required, self.height_required = 0, 0
        self.named_area_usage = {}
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                # m.weight [c_out, c_in, w, h]
                # crossbar input: ci x w x h
                # crossbar output: co
                # crossbar area: 
                #  - height = ci x w x h
                #  - width = co
                # nb queries: product_[d \in w, h] (d + 2 padding[d] - dilation[d] (kernel_size[d] - 1) - 1 / stride[d]) + 1 
                shape = m.weight.shape
                co, ci, w, h = shape
                self.width_required += co
                self.height_required += (ci * w * h)
                self.named_area_usage[n] = m.weight.numel()

            if isinstance(m, nn.Linear):
                # m.weight [f_out, f_in]
                # crossbar input: f_in
                # crossbar output: f_out
                # crossbar area:
                #   - height = f_in
                #   - width = f_out
                # nb queries: 1
                shape = m.weight.shape
                self.width_required += shape[0]
                self.height_required += shape[1]
                self.named_area_usage[n] = m.weight.numel()

        ###
        # Space-saving first: height-wise, assume we fill the unused rows with 0 at the input to have a more compact representation
        ###
        w_required = math.ceil(self.width_required / crossbar_size)
        h_required = math.ceil(self.height_required / crossbar_size)
        crossbar_required = max(w_required, h_required)
        usage_ratio = (self.width_required * self.height_required) / (crossbar_required * crossbar_size ** 2)
        print('Minimum width required ', self.width_required)
        print('Minimum height required ', self.height_required)
        print('Crossbar required ', crossbar_required)
        print('Usage ratio ', usage_ratio)
        print(self.named_area_usage)


if __name__ == '__main__':
    from thrifty import *
    model = thrifty18(1000, False, 10)
    Memristorage(model)