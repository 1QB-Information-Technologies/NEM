import math
import torch

import utils
from nqs_models.NeuralQuantumState import NeuralQuantumState


class LearnedPositionalEncoding(torch.nn.Module):
    def __init__(self, num_internal_sites, dim):
        """Represents a learned positional encoding.
        Args:
            num_internal_sites: number of internal sites = num_sites + 1
            dim: dimension of the output
        """
        super(LearnedPositionalEncoding, self).__init__()
        self.num_internal_sites = num_internal_sites
        self.dim = dim
        self.encodings = torch.nn.Parameter(
            torch.randn((num_internal_sites, dim)), requires_grad=True)

    def forward(self, batch_size, site=None):
        """
        Args:
            batch_size (int):
            site (int or None): if None, return encodings of all sites.
            Else just for site.
        Returns:
            torch float tensor of shape (batch_size, 1) or
            (batch_size, num_internal_sites)
        """
        if site is None:
            return self.encodings[None, :, :].expand((  batch_size,
                                                        self.num_internal_sites,
                                                        self.dim))
        else:
            return self.encodings[None, site:site+1, :].expand((batch_size,
                                                                1,
                                                                self.dim))


class FCLayer(torch.nn.Module):
    def __init__(self, dim, dropout=0.0):
        """Fully connected square linear layer.
        
        Guided by the dropout from
        https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py

        Args:
            dim (int): size of hidden dimension
            dropout (float): dropout rate. 0 means no dropout
        """
        super(FCLayer, self).__init__()
        if dropout > 0:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.Dropout(dropout))
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dim, dim))

    def forward(self, x):
        return self.net(x)



class GTrXLLayerWrapper(torch.nn.Module):
    def __init__(   self,
                    layer,
                    dim,
                    ):
        """Wraps a dimension preserving layer with layernorm, skip connection
        and gating. Following https://arxiv.org/pdf/1910.06764.pdf.
        Implemented gating adds skip connection and FF connection
        Args:
            layer: a Module, whose output is of the same dimension as the input.
                Should not modify its input
            dim (int): size of the hidden dimension, last dimension of the
                inputs and outputs

        """
        super(GTrXLLayerWrapper, self).__init__()

        self.gating = lambda x, y: x + y
        self.layernorm = torch.nn.LayerNorm([dim])
        self.layer = layer

    def forward(self, h, mem=None):
        if mem is None:
            x_skip = h
            y = self.layernorm(h)
            y = self.layer(y)
            return self.gating(x_skip, y.relu())
        else:
            x_skip = h
            y = self.layernorm(h)
            y, new_mem = self.layer(y, mem)
            return self.gating(x_skip, y.relu()), new_mem


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self,
                 num_internal_sites,
                 num_heads,
                 dim,
                 scale_attention_scores=False):
        """Multi-Head self attention layer. 
        
        Implementation guided by
        https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
        
        Args:
            num_internal_sites (int): num_sites + 1, includes start token site
            num_heads (int): number of attention heads
            dim (int): internal dimension. Should be divisible by num_heads.
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.num_internal_sites = num_internal_sites
        self.dim = dim
        self.scale_attention_scores = scale_attention_scores
        self.qvk_net = torch.nn.Linear(dim, 3*dim, bias=False)
        self.o_net = torch.nn.Linear(dim, dim, bias=False)

        mask = float('-inf') * torch.ones(( num_internal_sites,
                                            num_internal_sites),
                                            requires_grad=False)
        mask = mask.triu(1)
        self.register_buffer('mask', mask)

    def forward(self, h, mem=None):
        """
        The forward pass can run in one of two modes: if mem is None, run masked
        multi-head self attention for all sites. If mem is not None, run it just
        for the first site not contained in mem.

        if mem is None:
        Args:
            h: torch tensor (batch_size, num_internal_sites, dim), input activations

        Returns: torch tensor (batch_size, num_internal_sites, dim) with output activations

        if mem is not None:
        Args:
            h: torch tensor (batch_size, 1, dim): input activation at present site
            mem: list [k, v] with keys, values from past sites. List of two empty
                tensors if running for first site
        Returns: tuple (out, new_mem)
            out: torch tensor (batch_size, 1, dim): output activation at present site
            new_mem: list [k, v] with keys, values up to including present site

        """
        batch_size = h.shape[0]
        if mem is None:
            # b: batch. s: present site. t: past site. h:head. i:hidden
            qvk = self.qvk_net(h).reshape(( batch_size,
                                            self.num_internal_sites,
                                            3,
                                            self.num_heads,
                                            self.dim//self.num_heads))

            q, v, k = qvk.unbind(dim=2)

            attn_scores = torch.einsum('bshi,bthi->bsth', [q, k])
            attn_scores += self.mask[None, :, :, None]
            if self.scale_attention_scores:
                attn_scores /= math.sqrt(self.dim)
            attn_weights = attn_scores.softmax(dim=2)
            weighted_v = torch.einsum('bsth,bthi->bshi', [attn_weights, v])

            weighted_v = weighted_v.reshape((batch_size, self.num_internal_sites, self.dim))
            out = self.o_net(weighted_v)
            return out
        else:
            k_past, v_past = mem
            qvk_present = self.qvk_net(h).reshape(( batch_size,
                                                    1,
                                                    3,
                                                    self.num_heads,
                                                    self.dim//self.num_heads))

            q_present, v_present, k_present = qvk_present.unbind(dim=2)
            k = torch.cat([k_past, k_present], dim=1)
            v = torch.cat([v_past, v_present], dim=1)

            attn_scores_present = torch.einsum('bshi,bthi->bsth', [q_present, k])
            if self.scale_attention_scores:
                attn_scores_present /= math.sqrt(self.dim)
            attn_weights_present = attn_scores_present.softmax(dim=2)
            weighted_v_present = torch.einsum('bsth,bthi->bshi', [attn_weights_present, v])
            weighted_v_present = weighted_v_present.reshape((batch_size, 1, self.dim))
            out = self.o_net(weighted_v_present)
            new_mem = [k, v]
            return out, new_mem


class TransformerLayer(torch.nn.Module):
    def __init__(self,
                 num_internal_sites,
                 dim,
                 num_heads,
                 dropout=0.0,
                 scale_attention_scores=False):
        """Constructs a MultiHeadSelfAttention layer and an FCLayer,
        wraps both with GTrXLLayerWrapper and concatenates them.
        Following https://arxiv.org/pdf/1910.06764.pdf

        Args:
            num_internal_sites (int): number of sites, including start token site
            dim (int): hidden dimension
            num_heads (int): number of heads for multi head attention
            dropout (float): dropout rate
        """
        super(TransformerLayer, self).__init__()
        attn = MultiHeadSelfAttention(num_internal_sites=num_internal_sites,
                                      num_heads=num_heads,
                                      dim=dim,
                                      scale_attention_scores=scale_attention_scores)
        self.attn = GTrXLLayerWrapper(attn, dim)
        fc = FCLayer(dim=dim, dropout=dropout)
        self.fc = GTrXLLayerWrapper(fc, dim)

    def forward(self, h, mem=None):
        """
        The forward pass can run in one of two modes: if mem is None, run for all
        sites. If mem is not None, run it just for the first site not contained in mem.

        if mem is None:
        Args:
            h: torch tensor (batch_size, num_internal_sites, dim), input activations
        Returns: torch tensor (batch_size, num_internal_sites, dim) with output activations

        if mem is not None:
        Args:
            h: torch tensor (batch_size, 1, dim): input activation at present site
            mem: list [k, v] with keys, values from past sites. List of two empty
                tensors if running for first site.
        Returns: tuple (out, new_mem)
            out: torch tensor (batch_size, 1, dim): output activation at present site
            new_mem: list [k, v] with keys, values up to including present site

        """
        if mem is None:
            return self.fc(self.attn(h))
        else:
            attn_output_present, new_mem = self.attn(h, mem=mem)
            output_present = self.fc(attn_output_present)
            return output_present, new_mem


class Transformer(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 num_internal_sites,
                 dim,
                 num_heads,
                 dropout=0.0,
                 scale_attention_scores=False):
        """A Transformer with masked self-attention.
        Following the one in https://arxiv.org/pdf/1910.06764.pdf in some aspects.

        Args:
            num_layers (int): number of transformer layers
            num_internal_sites (int): number of sites including start token site
            dim (int): internal dimension
            num_heads (int): number of heads of MultiHeadSelfAttention components
            dropout (float): dropout rate for FC layers
        """
        super(Transformer, self).__init__()

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                TransformerLayer(
                    num_internal_sites=num_internal_sites,
                    dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    scale_attention_scores=scale_attention_scores
                )
            )
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, h, extra_encodings, mem=None):
        """This has two modes: if mem is None, run for all sites. In this case:
        Args:
            h: the input to the first layer, shape (batch_size, num_internal_sites, hidden)
            extra_encodings: shape (batch_size, num_internal_sites, hidden), gets
                added to the input state at every layer. This is where positional
                encodings and parameter embeddings will go

        Returns:
            torch tensor (batch_size, num_internal_sites, hidden), output of
            the last layer

        If mem is not None:

        Args:
            h: the input to the first layer at present site, (batch_size, 1, hidden)
            extra_encodings: shape (batch_size, 1, hidden); gets added to input at
                every layer
            mem: A list of mem which get passed to each layer. If running for
                first site, pass []
        Returns: (output, new_mem)
            output: (batch_size, 1, hidden). Output of last layer at present site
            new_mem: A list of mem, to be passed at the next step.
        """
        if mem is None:
            for layer in self.layers:
                h = layer(h+extra_encodings)
            return h
        else:
            if type(mem) is list and not mem:  # mem is empty list
                for _ in self.layers:
                    mem.append([torch.tensor([], device=h.device, dtype=h.dtype),
                                torch.tensor([], device=h.device, dtype=h.dtype)])
            for i, layer in enumerate(self.layers):
                h, mem[i] = layer(h+extra_encodings, mem=mem[i])
            return h, mem


class TransformerWF(NeuralQuantumState):
    def __init__(self,
                 num_sites,
                 num_layers,
                 internal_dimension,
                 num_heads,
                 dropout,
                 phase_mode=2,
                 scale_attention_scores=False):
        """ Transformer wavefunction.
        Args:
            num_sites (int): number of sites (not including start token site,
                that gets added here)
            num_layers (int): number of transformer layers
            internal_dimension (int): internal dimension
            num_heads (int): number of heads of MultiHeadSelfAttention components
            dropout (float): dropout rate for FC layers
            phase_mode (int): how to deal with the phase.
                0: no phase
                1: project transformer output with vector (shared over sites),
                    add results
                2: concatenate transformer outputs over sites, project with a
                    big learned vector
        """
        super(TransformerWF, self).__init__()
        assert phase_mode in range(3)
        self.num_sites = num_sites
        self.phase_mode = phase_mode
        self.scale_attention_scores = scale_attention_scores
        self.embedder = torch.nn.Embedding(2, internal_dimension)
        self.pos_encoder = LearnedPositionalEncoding(num_sites+1, internal_dimension)

        self.transformer = Transformer(
            num_layers=num_layers,
            num_internal_sites=num_sites+1,
            dim=internal_dimension,
            num_heads=num_heads,
            dropout=dropout,
            scale_attention_scores=scale_attention_scores
        )

        self.amplitude_head = torch.nn.Linear(internal_dimension, 1)
        if phase_mode == 1:
            self.phase_head = torch.nn.Linear(internal_dimension, 1)
        elif phase_mode == 2:
            self.phase_head = torch.nn.Linear(internal_dimension*(num_sites+1), 1)

    def forward(self,
                samples,
                return_conditional_logprobs=False,
                return_transformer_output=False):
        """For internal use, use amplitudes and sample instead
        Args:
            samples: uint8 tensor of shape (..., num_sites)

        Returns:
            log-amplitudes, Complex of shape (...,)
        """

        input_shape = samples.shape[:-1]
        internal_samples = samples.reshape((-1, self.num_sites))
        internal_samples = torch.cat(
            [torch.zeros_like(internal_samples[:, :1]), internal_samples], dim=-1)

        internal_batch_size = internal_samples.shape[0]

        h = self.embedder(internal_samples.long())
        extra_encodings = self.pos_encoder(internal_batch_size).clone()

        transformer_output = self.transformer(h, extra_encodings)

        logits = self.amplitude_head(transformer_output[:, :-1, :]).reshape((-1, self.num_sites))

        conditional_logprobs = torch.nn.functional.logsigmoid(
            (2 * internal_samples[:, 1:].float() - 1) * logits
        )
        logmoduli = .5 * conditional_logprobs.sum(dim=-1)

        if self.phase_mode == 0:
            phases = torch.zeros_like(logmoduli)
        elif self.phase_mode == 1:
            
            phases_per_site = self.phase_head(
                                transformer_output[:, 1:, :]
                                ).reshape((-1, self.num_sites))

            phases = phases_per_site.sum(dim=-1)
        elif self.phase_mode == 2:

            phases = self.phase_head(
                                transformer_output.reshape(
                                    (transformer_output.shape[0], -1))).flatten()

        logamplitudes = utils.Complex(logmoduli, phases).reshape(input_shape)
        out = [logamplitudes]
        if return_conditional_logprobs:
            conditional_logprobs = conditional_logprobs.reshape( input_shape + (self.num_sites,))
            out = out+[conditional_logprobs]
        if return_transformer_output:
            transformer_output = transformer_output.reshape(
                input_shape + (self.num_sites+1, -1))
            out = out + [transformer_output]

        return out

    def amplitudes(self,
                   samples,
                   return_polar=False,
                   return_conditional_logprobs=False,
                   return_transformer_output=False):
        """

        Args:
            samples: torch uint8 tensor of shape (..., num_sites)
            return_polar (boolean): Whether to return amps (False) or log_amps (True)

        Returns:
            amplitudes or logamplitudes, Complex of shape (...,)

        """

        out = self(samples,
                   return_conditional_logprobs=return_conditional_logprobs,
                   return_transformer_output=return_transformer_output)

        if not return_polar:
            out[0] = out[0].exp()

        if len(out)==1:
            return out[0]
        else:
            return out

    def sample(self, num_samples):

        extra_encodings = self.pos_encoder(num_samples).clone()


        internal_samples = torch.zeros((num_samples, self.num_sites + 1),
                                       dtype=torch.uint8,
                                       device=next(self.parameters()).device)
        mem = []
        for i in range(self.num_sites):
            transformer_input = self.embedder(internal_samples[:, i:i+1].long())
            transformer_output, mem = self.transformer( transformer_input,
                                                        extra_encodings[:, i:i+1, :],
                                                        mem=mem)

            logits = self.amplitude_head(transformer_output).flatten()

            dist = torch.distributions.Bernoulli(logits=logits)
            next_bit = dist.sample().to(torch.uint8)
            internal_samples[:, i+1] = next_bit

        return internal_samples[:, 1:]
