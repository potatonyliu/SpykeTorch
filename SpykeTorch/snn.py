import torch
import torch.nn as nn
import torch.nn.functional as fn
from . import functional as sf
from torch.nn.parameter import Parameter
from .utils import to_pair

class Convolution(nn.Module):
    r"""Performs a 2D convolution over an input spike-wave composed of several input
    planes. Current version only supports stride of 1 with no padding.

    The input is a 4D tensor with the size :math:`(T, C_{{in}}, H_{{in}}, W_{{in}})` and the crresponsing output
    is of size :math:`(T, C_{{out}}, H_{{out}}, W_{{out}})`, 
    where :math:`T` is the number of time steps, :math:`C` is the number of feature maps (channels), and
    :math:`H`, and :math:`W` are the hight and width of the input/output planes.

    * :attr:`in_channels` controls the number of input planes (channels/feature maps).

    * :attr:`out_channels` controls the number of feature maps in the current layer.

    * :attr:`kernel_size` controls the size of the convolution kernel. It can be a single integer or a tuple of two integers.

    * :attr:`weight_mean` controls the mean of the normal distribution used for initial random weights.

    * :attr:`weight_std` controls the standard deviation of the normal distribution used for initial random weights.

    .. note::

        Since this version of convolution does not support padding, it is the user responsibility to add proper padding
        on the input before applying convolution.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        weight_mean (float, optional): Mean of the initial random weights. Default: 0.8
        weight_std (float, optional): Standard deviation of the initial random weights. Default: 0.02
    """
    def __init__(self, in_channels, out_channels, kernel_size, weight_mean=0.8, weight_std=0.02):
        super(Convolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_pair(kernel_size)
        #self.weight_mean = weight_mean
        #self.weight_std = weight_std

        # For future use
        self.stride = 1
        self.bias = None
        self.dilation = 1
        self.groups = 1
        self.padding = 0

        # Parameters
        self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        self.weight.requires_grad_(False) # We do not use gradients
        self.reset_weight(weight_mean, weight_std)

        # New parameters for STDP
        self.decay = 0.95
        self.threshold = 10.0
        self.reset = 0.0

    def reset_weight(self, weight_mean=0.8, weight_std=0.02):
        """Resets weights to random values based on a normal distribution.

        Args:
            weight_mean (float, optional): Mean of the random weights. Default: 0.8
            weight_std (float, optional): Standard deviation of the random weights. Default: 0.02
        """
        self.weight.normal_(weight_mean, weight_std)

    def load_weight(self, target):
        """Loads weights with the target tensor.

        Args:
            target (Tensor=): The target tensor.
        """
        self.weight.copy_(target)	

    def forward(self, input):
        return fn.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Pooling(nn.Module):
    r"""Performs a 2D max-pooling over an input signal (spike-wave or potentials) composed of several input
    planes.

    .. note::

        Regarding the structure of the spike-wave tensors, application of max-pooling over spike-wave tensors results
        in propagation of the earliest spike within each pooling window.

    The input is a 4D tensor with the size :math:`(T, C, H_{{in}}, W_{{in}})` and the crresponsing output
    is of size :math:`(T, C, H_{{out}}, W_{{out}})`, 
    where :math:`T` is the number of time steps, :math:`C` is the number of feature maps (channels), and
    :math:`H`, and :math:`W` are the hight and width of the input/output planes.

    * :attr:`kernel_size` controls the size of the pooling window. It can be a single integer or a tuple of two integers.

    * :attr:`stride` controls the stride of the pooling. It can be a single integer or a tuple of two integers. If the value is None, it does pooling with full stride.

    * :attr:`padding` controls the amount of padding. It can be a single integer or a tuple of two integers.

    Args:
        kernel_size (int or tuple): Size of the pooling window
        stride (int or tuple, optional): Stride of the pooling window. Default: None
        padding (int or tuple, optional): Size of the padding. Default: 0
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super(Pooling, self).__init__()
        self.kernel_size = to_pair(kernel_size)
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = to_pair(stride)
        self.padding = to_pair(padding)

        # For future use
        self.dilation = 1
        self.return_indices = False
        self.ceil_mode = False
        
    def forward(self, input):
        return sf.pooling(input, self.kernel_size, self.stride, self.padding)

class STDP(nn.Module):
    r"""Performs STDP learning rule over synapses of a convolutional layer based on the following formulation:

    The following equations allows the use of `torch.cumsum()` to compute pre-synaptic spikes with decay.

    .. math::
        \sum_{i=0}^t S(i) \cdot \gamma^{t-i}=\gamma^{t}\sum_{i=0}^t S(i) \cdot \gamma^{-i}

    where :math:`i` and :math:`j` refer to the post- and pre-synaptic neurons, respectively,
    :math:`\Delta w_{ij}` is the amount of weight change for the synapse connecting the two neurons,
    and :math:`a_{LTP}`, and :math:`a_{LTD}` scale the magnitude of weight change. Besides,
    :math:`\left(W_{ij}-W_{LB}\right)\times \left(W_{UP}-W_{ij}\right)` is a stabilizer term which
    slowes down the weight change when the synaptic weight is close to the weight's lower (:math:`W_{LB}`)
    and upper (:math:`W_{UB}`) bounds.

    To create a STDP object, you need to provide:

    * :attr:`conv_layer`: The convolutional layer on which the STDP should be applied.

    * :attr:`learning_rate`: (:math:`a_{LTP}`, :math:`a_{LTD}`) rates. A single pair of floats or a list of pairs of floats. Each feature map has its own learning rates.

    * :attr:`use_stabilizer`: Turns the stabilizer term on or off.

    * :attr:`lower_bound` and :attr:`upper_bound`: Control the range of weights.

    To apply STDP for a particular stimulus, you need to provide:
    
    * :attr:`input_spikes` and :attr:`potentials` that are the input spike-wave and corresponding potentials, respectively.

    * :attr:`output_spikes` that is the output spike-wave.

    * :attr:`winners` or :attr:`kwta` to find winners based on the earliest spike then the maximum potential.

    * :attr:`inhibition_radius` to inhibit surrounding neurons (in all feature maps) within a particular radius.

    Args:
        conv_layer (snn.Convolution): Reference convolutional layer.
        learning_rate (tuple of floats or list of tuples of floats): (LTP, LTD) rates for STDP.
        use_stabilizer (boolean, optional): Turning stabilizer term on or off. Default: True
        lower_bound (float, optional): Lower bound of the weight range. Default: 0
        upper_bound (float, optional): Upper bound of the weight range. Default: 1
    """
    def __init__(self, conv_layer, learning_rate, timesteps, use_stabilizer = True, lower_bound = 0, upper_bound = 1):
        super(STDP, self).__init__()
        self.conv_layer = conv_layer
        if isinstance(learning_rate, list):
            self.learning_rate = learning_rate
        else:
            self.learning_rate = [learning_rate] * conv_layer.out_channels
        for i in range(conv_layer.out_channels):
            self.learning_rate[i] = (Parameter(torch.tensor([self.learning_rate[i][0]])),
                            Parameter(torch.tensor([self.learning_rate[i][1]])))
            self.register_parameter('ltp_' + str(i), self.learning_rate[i][0])
            self.register_parameter('ltd_' + str(i), self.learning_rate[i][1])
            self.learning_rate[i][0].requires_grad_(False)
            self.learning_rate[i][1].requires_grad_(False)
        self.use_stabilizer = use_stabilizer
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # decay matrices
        t = torch.arange(timesteps)
        self.register_buffer('gamma_t',torch.pow(conv_layer.decay, t).view(timesteps, 1, 1, 1))
        self.register_buffer('gamma_neg_t',torch.pow(conv_layer.decay, -t).view(timesteps, 1, 1, 1))




    # simple STDP rule
    # gets prepost pairings, winners, weights, and learning rates (all shoud be tensors)
    def forward(self, input_spikes, potentials, output_spikes, winners, kwta = 1):
        if len(winners) == 0:
            raise ValueError("Warning: No winner found!")
        # Get dimentions of the previous layer kernel
        kernel_size = self.conv_layer.kernel_size
        dilation = self.conv_layer.dilation
        padding = self.conv_layer.padding
        stride = self.conv_layer.stride
        kH, kW = self.conv_layer.kernel_size
        weight_updates = torch.zeros_like(self.conv_layer.weight)

        kH_radius = (kH-1)//2
        kW_radius = (kW-1)//2

        for (t, f, h, w) in winners:
            # Center of receptive field in input coordinate
            h_in = h * stride - padding + kH_radius
            w_in = w * stride - padding + kW_radius
            presynaptic_spikes = input_spikes[:t,:,h_in-kH_radius:h_in+kH_radius+1,w_in-kW_radius:w_in+kW_radius+1]
            postsynaptic_spikes = input_spikes[t+1:,:,h_in-kH_radius:h_in+kH_radius+1,w_in-kW_radius:w_in+kW_radius+1]
            weight_updates[f] += self.get_buffer('gamma_t')[t] * torch.sum(presynaptic_spikes * self.get_buffer('gamma_neg_t')[:t],dim=0) * self.learning_rate[f][0] + self.get_buffer('gamma_neg_t')[t] * torch.sum(postsynaptic_spikes * self.get_buffer('gamma_t')[t+1:],dim=0) * self.learning_rate[f][1]

        self.conv_layer.weight += weight_updates * ((self.conv_layer.weight-self.lower_bound) * (self.upper_bound-self.conv_layer.weight) if self.use_stabilizer else 1)
        self.conv_layer.weight.clamp_(self.lower_bound, self.upper_bound)

    def update_learning_rate(self, feature, ap, an):
        r"""Updates learning rate for a specific feature map.

        Args:
            feature (int): The target feature.
            ap (float): LTP rate.
            an (float): LTD rate.
        """
        self.learning_rate[feature][0][0] = ap
        self.learning_rate[feature][1][0] = an

    def update_all_learning_rate(self, ap, an):
        r"""Updates learning rates of all the feature maps to a same value.

        Args:
            ap (float): LTP rate.
            an (float): LTD rate.
        """
        for feature in range(self.conv_layer.out_channels):
            self.learning_rate[feature][0][0] = ap
            self.learning_rate[feature][1][0] = an
