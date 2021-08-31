import torch
import numpy as np

class Complex():
    @staticmethod
    def einsum(order, m1, m2):
        '''
        only supports two factors
        '''
        real = torch.einsum(order, m1.real, m2.real) - torch.einsum(order, m1.imag, m2.imag)
        imag = torch.einsum(order, m1.real, m2.imag) + torch.einsum(order, m1.imag, m2.real)
        return Complex(real, imag)

    @staticmethod
    def stack(tensors, **kwargs):
        real = torch.stack([t.real for t in tensors], **kwargs)
        imag = torch.stack([t.imag for t in tensors], **kwargs)
        return Complex(real, imag)
    @staticmethod
    def cat(tensors, *args, **kwargs):
        real = torch.cat([t.real for t in tensors], *args, **kwargs)
        imag = torch.cat([t.imag for t in tensors], *args, **kwargs)
        return Complex(real, imag)
    def __init__(self, real, imag=None):
        if type(real) == np.ndarray: #and real.dtype == np.complex:
            self.real = torch.tensor(real.real)
            self.imag = torch.tensor(real.imag)
        else:
            self.real = real
            if imag is not None:
                self.imag = imag
            else:
                self.imag = torch.zeros_like(real)

    def __getitem__(self, item):
        real = self.real[item]
        imag = self.imag[item]
        return Complex(real, imag)

    def __str__(self):
        return f'real: {self.real} imag: {self.imag}'

    def __len__(self):
        return len(self.real)

    def __add__(self, other):
        real = self.real + other.real
        imag = self.imag + other.imag
        return Complex(real, imag)

    def __sub__(self, other):
        real = self.real - other.real
        imag = self.imag - other.imag
        return Complex(real, imag)

    def __truediv__(self, other):
        denom = (other.real.pow(2) + other.imag.pow(2) + 1e-10)
        real = self.real * other.real + self.imag * other.imag
        imag = - self.real * other.imag + self.imag * other.real
        return Complex(real/denom, imag/denom)

    def __mul__(self, other):
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        return Complex(real, imag)

    def to(self, *args, **kwargs):
        real = self.real.to(*args, **kwargs)
        imag = self.imag.to(*args, **kwargs)
        return Complex(real, imag)

    def reshape(self, *args, **kwargs):
        real = self.real.reshape(*args, **kwargs)
        imag = self.imag.reshape(*args, **kwargs)
        return Complex(real, imag)

    def conjugate(self):
        return Complex(self.real, -self.imag)

    def sum(self, dim=None):
        real = self.real.sum(dim=dim)
        imag = self.imag.sum(dim=dim)
        return Complex(real, imag)

    def exp(self):
        real = self.real.exp() * self.imag.cos()
        imag = self.real.exp() * self.imag.sin()
        return Complex(real, imag)

    def log(self):
        real = .5*(self.real.pow(2) + self.imag.pow(2)).log()
        imag = torch.atan2(self.imag, self.real)
        return Complex(real, imag)

    def norm2(self):
        return self.real.pow(2) + self.imag.pow(2)

    def abs(self):
        return (self.real.pow(2) + self.imag.pow(2)).sqrt()

    def mean(self, *args, **kwargs):
        real = self.real.mean(*args, **kwargs)
        imag = self.imag.mean(*args, **kwargs)
        return Complex(real, imag)

    def detach(self):
        real = self.real.detach()
        imag = self.imag.detach()
        return Complex(real,imag)
        
    def numpy(self):
        return self.real.numpy() + 1j*self.imag.numpy()

    @property
    def shape(self):
        return self.real.shape
