import torch
from icecream import ic

IMPORTANCE_REGISTRY = {}

def register_importance(type_name):
    """
    Decorator to register an importance computation function by type name.
    
    Args:
        type_name (str): Key name to register the function
    """
    def decorator(func):
        IMPORTANCE_REGISTRY[type_name] = func
        return func
    return decorator

def get_importance(type_name, weights=None, activations=None, **kwargs):
    """
    Dispatch importance computation based on registered type.
    
    Args:
        type_name (str): one of ['weight', 'activation', 'wanda', 'entropy']
        weights (Tensor): required if method uses weights
        activations (Tensor): required if method uses activations
        **kwargs: additional arguments passed to the method (e.g. norm)
    
    Returns:
        Tensor: importance scores per channel
    """
    if type_name not in IMPORTANCE_REGISTRY:
        raise ValueError(f"Unknown importance type: {type_name}")
    return IMPORTANCE_REGISTRY[type_name](weights=weights, activations=activations, **kwargs)

@register_importance('weight')
def compute_weight_importance(weights, activations=None):
    """
    Compute weight-based importance over batched tasks and layers.

    Args:
        weights (Tensor): shape (k, m, l, out_dim, in_dim)

    Returns:
        Tensor: shape (l, in_dim), average importance per layer and input channel
    """
    # abs().sum over output dim, then mean over tasks and samples
    importance = weights.abs().sum(dim=-2).mean(dim=(0, 1))  # shape: (l, in_dim)
    return importance


@register_importance('activation')
def compute_activation_importance(activations, weights=None, norm=2):
    """
    Compute activation-based importance over batched tasks and layers.

    Args:
        activations (Tensor): shape (k, m, l, in_dim)
        norm (int): p-norm value (default: 2)

    Returns:
        Tensor: shape (l, in_dim), average norm importance
    """
    # norm over batch dim, then mean over tasks and samples
    norm_vals = torch.norm(activations, dim=1, p=norm)  # shape: (k, l, in_dim)
    importance = norm_vals.mean(dim=0)  # shape: (l, in_dim)
    return importance


@register_importance('wanda')
def compute_wanda_importance(weights, activations, norm=2):
    """
    Compute WANDA-style importance: weight × activation norm.

    Args:
        weights (Tensor): shape (k, m, l, out_dim, in_dim)
        activations (Tensor): shape (k, m, l, in_dim)
        norm (int): p-norm for activation (default: 2)

    Returns:
        Tensor: shape (l, in_dim)
    """
    w_imp = compute_weight_importance(weights)          # shape: (l, in_dim)
    a_imp = compute_activation_importance(activations, norm)  # shape: (l, in_dim)
    return w_imp * a_imp  # shape: (l, in_dim)

@register_importance('entropy')
def compute_entropy_importance(activations, weights=None, bins=30, gamma_=0.5, lambda_=0.5):
    """
    Compute channel importance for each layer based on entropy across tasks.

    Args:
        activations (torch.Tensor): shape (k, m, l, c)
            k: number of tasks
            m: number of samples per task
            l: number of layers
            c: number of channels per layer
        bins (int): number of histogram bins used to estimate entropy
        gamma_ (float): weight for mean entropy
        lambda_ (float): weight for entropy std across tasks

    Returns:
        importance (torch.Tensor): shape (l, c)
            Importance score per channel for each layer.
    """

    def compute_entropy_per_channel(act_task_layer):
        """
        Compute entropy for each channel in a single task & single layer.

        Args:
            act_task_layer (torch.Tensor): shape (m, c)

        Returns:
            entropy_per_channel (torch.Tensor): shape (c,)
        """
        m, c = act_task_layer.shape
        entropy_list = []
        for i in range(c):
            values = act_task_layer[:, i].cpu()
            hist = torch.histc(values, bins=bins, min=values.min().item(), max=values.max().item())
            probs = hist / (hist.sum() + 1e-8)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            entropy_list.append(entropy)
        return torch.stack(entropy_list)

    k, m, l, c = activations.shape
    # shape: (k, l, c) -- entropy per task, per layer, per channel
    entropy_matrix = torch.stack([
        torch.stack([compute_entropy_per_channel(activations[t, :, lyr]) for lyr in range(l)])
        for t in range(k)
    ])  # shape: (k, l, c)

    mean_entropy = entropy_matrix.mean(dim=0)  # shape: (l, c)
    std_entropy = entropy_matrix.std(dim=0)    # shape: (l, c)

    importance = gamma_ * mean_entropy - lambda_ * std_entropy  # shape: (l, c)
    return importance

if __name__ == '__main__':
    act = torch.randn(1, 2, 28, 8960)
    weights = torch.randn(1, 2, 28, 1536, 8960)
    multi_act = torch.randn(3, 2, 28, 8960)
    importance_e = get_importance('entropy', activations=multi_act)
    importance_a = get_importance('activation', activations=act)
    importance_w = get_importance('weight', weights=weights)
    importance_wanda = get_importance('wanda', activations=act, weights=weights)
    ic(importance_e.shape, importance_a.shape, importance_w.shape, importance_wanda.shape)
    # ic(importance_a.shape)