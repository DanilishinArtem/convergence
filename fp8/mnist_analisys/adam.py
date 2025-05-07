import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from convTool import ConvTool

import torch

def bitflip_tensor(tensor: torch.Tensor, bit_index: int, element_index: int = 0):
    assert tensor.dtype == torch.float32 or tensor.dtype == torch.int32, "Только float32 или int32 поддерживается"
    flat_tensor = tensor.flatten().clone()
    if element_index >= flat_tensor.numel():
        raise IndexError("element_index вне диапазона тензора")
    if tensor.dtype == torch.float32:
        int_view = flat_tensor.view(torch.int32)
    else:
        int_view = flat_tensor
    mask = 1 << (31 - bit_index)
    int_view[element_index] ^= mask
    result = int_view.view(tensor.dtype).reshape(tensor.shape)
    return result


class ScaledAdam(Optimizer):
    def __init__(self, params, writer, lr, beta=0.9, eps=1e-8, rebound='constant', warmup=500, init_lr=None, weight_decay=0, weight_decay_type=None):
        self.layer = None
        self.writer = writer
        self.counter = 0
        self.convTool = ConvTool(exp_coeff=0.999, writer=self.writer)
        self.low_precision = False
        self.layers_name = [
            'conv1_weight',
            'conv1_bias',
            'conv2_weight',
            'conv2_bias',
            'lin1_weight',
            'lin1_bias',
            'lin2_weight',
            'lin2_bias',
        ]

        if not 0.0 < lr:
            raise ValueError("Invalid learning rate value: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if rebound not in ['constant', 'belief']:
            raise ValueError("Invalid recitifed bound: {}".format(rebound))
        if not 0.0 <= warmup:
            raise ValueError("Invalid warmup updates: {}".format(warmup))
        if init_lr is None:
            init_lr = lr / 1000
        if not 0.0 <= init_lr <= lr:
            raise ValueError("Invalid initial learning rate: {}".format(init_lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if weight_decay_type is None:
            weight_decay_type = 'L2' if rebound == 'constant' else 'decoupled'
        if weight_decay_type not in ['L2', 'decoupled', 'stable']:
            raise ValueError("Invalid weight decay type: {}".format(weight_decay_type))

        defaults = dict(lr=lr, beta=beta, eps=eps, rebound=rebound,
                        warmup=warmup, init_lr=init_lr, base_lr=lr,
                        weight_decay=weight_decay, weight_decay_type=weight_decay_type)
        super(ScaledAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ScaledAdam, self).__setstate__(state)

    def tensor_to_fp8(self, tensor, exponent_bits=4, mantissa_bits=3):
        max_exponent = 2 ** (exponent_bits - 1) - 1
        min_exponent = -max_exponent + 1
        max_mantissa = 2 ** mantissa_bits - 1
        scale = 2.0 ** min_exponent
        tensor_scaled = tensor / scale
        tensor_quantized = torch.round(tensor_scaled * max_mantissa) / max_mantissa
        tensor_fp8 = tensor_quantized * scale
        return tensor_fp8

    @torch.no_grad()
    def step(self, closure=None):
        self.layer = 0
        self.counter += 1
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)


                state['step'] += 1

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Atom does not support sparse gradients.')


                beta, eps = group['beta'], group['eps']
                adam_beta1, adam_beta2 = 0.9, 0.99
                bias_correction1 = 1 - adam_beta1 ** state['step']
                bias_correction2 = 1 - adam_beta2 ** state['step']
                step_size = group['lr'] / bias_correction1

                # Корректное направление обновления (Adam)
                # prev_grad = state['exp_avg'].clone()
                state['exp_avg'].mul_(adam_beta1).add_(grad, alpha=1-adam_beta1)
                state['exp_avg_sq'].mul_(adam_beta2).addcmul_(grad, grad, value=1-adam_beta2)

                # Quantization ----------------------------------------------------------------------------------------
                # e, m = 5, 2
                # if self.convTool.time_conv and self.convTool.time_conv < 600:
                #     self.low_precision = True
                #     state['exp_avg'] = self.tensor_to_fp8(state['exp_avg'], exponent_bits=e, mantissa_bits=m)

                # state['exp_avg'] = self.tensor_to_fp8(state['exp_avg'], exponent_bits=e, mantissa_bits=m)
                # state['exp_avg_sq'] = self.tensor_to_fp8(state['exp_avg_sq'], exponent_bits=5, mantissa_bits=10)
                # Quantization ----------------------------------------------------------------------------------------
                self.convTool.accumulateInfo(p.data.clone().flatten(), state['exp_avg_sq'].clone().flatten(), grad.data.clone().flatten())

                denom = (state['exp_avg_sq'].sqrt() / (bias_correction2**0.5)).add_(group['eps'])
                d_p = -step_size * state['exp_avg'] / denom

                p.data.add_(d_p)

                if self.layer == 0 and state['step'] == 500:# and state['step'] < 600:
                    source_val = p.data.view(-1)[0]
                    # new_val = bitflip_tensor(source_val, 2)
                    new_val = 10
                    print("[INFO] fault injected for layer {}, {} ---> {}".format(self.layer, source_val, new_val))
                    p.data.view(-1)[0] = new_val

                self.layer += 1

        self.convTool.calculate_T(step_size, group['eps'], state['step'], adam_beta1, adam_beta2, epsilon=1e-4)
        if self.low_precision:
            self.writer.add_scalar("Low precision", 1, state['step'])
        else:
            self.writer.add_scalar("Low precision", 0, state['step'])
        self.low_precision = False
        return loss
    

# https://arxiv.org/pdf/2412.05270
# https://arxiv.org/pdf/2009.13586