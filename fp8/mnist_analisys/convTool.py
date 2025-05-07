import torch
from config import Config as config

class ConvTool:
    def __init__(self, exp_coeff, writer = None):
        self.initialized = False
        self.writer = writer
        self.exp_coeff = exp_coeff

        self.initial_weights = torch.tensor([])
        self.current_weights = torch.tensor([])
        self.exp_avg_sq = torch.tensor([])
        self.grads = torch.tensor([])
        if config.device == 'cuda':
            self.initial_weights = self.initial_weights.cuda()
            self.current_weights = self.current_weights.cuda()
            self.exp_avg_sq = self.exp_avg_sq.cuda()
            self.grads = self.grads.cuda()

        self.time_conv = None

    def accumulateInfo(self, weights, exp_avg_sq, grads):
        if not self.initialized:
            self.initial_weights = torch.cat((self.initial_weights, weights))
        self.current_weights = torch.cat((self.current_weights, weights))
        self.exp_avg_sq = torch.cat((self.exp_avg_sq, exp_avg_sq))
        self.grads = torch.cat((self.grads, grads))

    def calculate_T(self, eta, eps, step, beta1, beta2, epsilon=1e-4, Sigma=0.0):
        self.initialized = True
        k1 = (1 - beta1) / 1e-10
        bias_correction = 1 - beta2 ** step
        H_sq = self.exp_avg_sq / bias_correction
        H = torch.sqrt(H_sq + eps)
        delta_sq = torch.mean(self.grads ** 2)
        delta_theta_sq_current = torch.var(self.current_weights)
        delta_theta_sq_initial = torch.var(self.initial_weights)

        numerator = eta * delta_sq * (2 + k1)
        denominator = (2 * k1 * H * torch.sqrt(delta_sq + H_sq * delta_theta_sq_current + Sigma)).mean()
        delta_theta_sq_st = numerator / denominator
        v_st = H_sq * delta_theta_sq_current + delta_sq
        sqrt_v_sq = torch.sqrt(torch.mean(v_st))
        H_avg = torch.mean(H)
        # log_numerator = torch.abs(delta_theta_sq_initial - torch.mean(delta_theta_sq_st))
        # log_denominator = torch.abs(epsilon - torch.mean(delta_theta_sq_st))
        log_numerator = delta_theta_sq_initial
        log_denominator = epsilon
        T = (sqrt_v_sq / (2 * eta * H_avg)) * torch.log(log_numerator / log_denominator)

        self.time_conv = T if self.time_conv == None else self.exp_coeff * self.time_conv + (1 - self.exp_coeff) * T

        self.current_weights.resize_(0)
        self.exp_avg_sq.resize_(0)
        self.grads.resize_(0)

        if self.writer:
            self.writer.add_scalar("Time of convergence", self.time_conv, step)