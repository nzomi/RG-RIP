import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import FFN_MAPPING_BY_MODEL_TYPE

class FFNAccessor:
    """
    A helper class to access and manipulate feed-forward network (FFN) layers
    in transformer models with different MLP naming conventions, such as InternLM2 and Qwen2.

    It supports:
    - Accessing gate, down, up linear layers in each transformer layer
    - Registering forward hooks to capture activations
    - Retrieving weights of specified linear layers
    - Applying channel pruning and replacing linear layers accordingly
    """

    def __init__(self, model, model_type):
        """
        Initialize the accessor.

        Args:
            model: The full transformer model instance.
            model_type (str): The model type string, e.g., "internlm2", "qwen2".
        """
        self.model = model
        self.model_type = model_type.lower()
        self.config = FFN_MAPPING_BY_MODEL_TYPE[self.model_type]
        self.prefix = self.config["prefix"]  # e.g., 'feed_forward' or 'mlp'
        self.name_map = self.config["name_map"]  # e.g., {'w1': 'gate', 'w2': 'down', 'w3': 'up'}
        self.reverse_map = {v: k for k, v in self.name_map.items()}  # Invert map for attribute lookup

    def get_layer_module(self, layer_idx, tag):
        """
        Get the specific Linear module (gate/down/up) from the transformer layer.

        Args:
            layer_idx (int): The index of the transformer layer.
            tag (str): One of 'gate', 'down', 'up' to specify which linear to get.

        Returns:
            nn.Linear: The requested Linear module.
        """
        layer = self.model.language_model.model.layers[layer_idx]
        block = getattr(layer, self.prefix)
        return getattr(block, self.reverse_map[tag])

    def iterate_layers(self):
        """
        Iterate through all transformer layers yielding (index, (gate, down, up) Linear modules).

        Yields:
            Tuple[int, Tuple[nn.Linear, nn.Linear, nn.Linear]]:
                Layer index and tuple of gate, down, up Linear modules.
        """
        for i in range(len(self.model.language_model.model.layers)):
            yield i, (
                self.get_layer_module(i, "gate"),
                self.get_layer_module(i, "down"),
                self.get_layer_module(i, "up"),
            )

    def register_hooks(self, get_activation_fn):
        """
        Register forward hooks on gate, down, and up Linear layers to record activations.

        Args:
            get_activation_fn (Callable): A function that returns a hook function, 
                typically like `get_activation(name)` that captures activation by name.
        """
        for i in range(len(self.model.language_model.model.layers)):
            for tag in ["gate", "down", "up"]:
                module = self.get_layer_module(i, tag)
                raw_name = self.reverse_map[tag]
                module.register_forward_hook(get_activation_fn(f"layer.{i}.{raw_name}"))

    def get_weights(self, tag):
        """
        Get weights from all layers for a specific Linear type (gate/down/up).

        Args:
            tag (str): One of 'gate', 'down', 'up'.

        Returns:
            List[Tensor]: List of weight tensors for the specified Linear layers.
        """
        return [self.get_layer_module(i, tag).weight for i in range(len(self.model.language_model.model.layers))]
    
    def compute_mlp_activate(self, activations, num_layers):
        """
        Compute MLP activations as SiLU(gate) * up dynamically.

        Args:
            activations (dict): Dict mapping activation names to tensors.
            num_layers (int): Number of layers.

        Returns:
            List of activation tensors.
        """
        gate_name = self.reverse_map["gate"]
        up_name = self.reverse_map["up"]

        assert f"layer.0.{gate_name}" in activations, "Activation hook did not fire!"

        act = []
        for i in range(num_layers):
            gate = activations[f"layer.{i}.{gate_name}"]
            up = activations[f"layer.{i}.{up_name}"]
            act.append(F.silu(gate) * up)
        return act

    def prune_linear_channel(self, linear_layer, selected_channel, prune_dim):
        """
        Prune a Linear layer by selecting a subset of input or output channels.

        Args:
            linear_layer (nn.Linear): The original linear layer to prune.
            selected_channel (Tensor): The indices of channels to keep.
            prune_dim (int): Dimension to prune, 0 for input channels, 1 for output channels.

        Returns:
            nn.Linear: A new Linear layer with pruned channels.
        """
        if prune_dim == 1:  # pruning output channels (e.g., down_proj)
            in_features = selected_channel.numel()
            out_features = linear_layer.out_features
        else:  # pruning input channels (e.g., up_proj or gate_proj)
            out_features = selected_channel.numel()
            in_features = linear_layer.in_features

        bias = linear_layer.bias is not None
        new_linear = nn.Linear(in_features, out_features, bias=bias)

        # Copy selected weights along prune_dim dimension
        new_linear.weight.data.copy_(
            torch.index_select(linear_layer.weight.data, prune_dim, selected_channel)
        )

        # Copy bias if present
        if bias:
            new_linear.bias.data.copy_(linear_layer.bias.data)

        return new_linear

    @torch.no_grad()
    def apply_mlp_prune(self, sample_idx):
        """
        Apply channel pruning to MLP layers (gate, down, up) in all transformer layers,
        replacing the original linear layers with pruned versions.

        Args:
            sample_idx (List[Tensor]): List of channel indices to keep for each layer.

        Returns:
            model: A deepcopy of the original model with pruned MLP layers replaced.
        """
        import copy
        self.model = copy.deepcopy(self.model)

        for layer_idx, (gate, down, up) in self.iterate_layers():
            new_down = self.prune_linear_channel(down, sample_idx[layer_idx], prune_dim=1)
            new_up = self.prune_linear_channel(up, sample_idx[layer_idx], prune_dim=0)
            new_gate = self.prune_linear_channel(gate, sample_idx[layer_idx], prune_dim=0)

            decoder_layer = self.model.language_model.model.layers[layer_idx]
            # Use model-specific assign function to set new Linear layers
            self.config["assign"](decoder_layer, new_gate, new_down, new_up)

        return self.model

