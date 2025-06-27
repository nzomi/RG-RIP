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
    weights: List[List[Tensor]] -> shape: (k tasks, m samples) each of shape (l, out, in)
    Returns: Tensor of shape (l, in)
    """
    k = len(weights)
    m = len(weights[0])
    l, out_dim, in_dim = weights[0][0].shape

    total = torch.zeros(l, in_dim, device=weights[0][0].device)
    for task_weights in weights:       # over k
        for sample in task_weights:    # over m
            total += sample.abs().sum(dim=1)  # sum over out_dim → shape: (l, in_dim)

    return total / (k * m)


@register_importance('activation')
def compute_activation_importance(activations, weights=None, norm=2):
    """
    activations: List[List[Tensor]] -> shape: (k, m) each of shape (l, in)
    Returns: Tensor of shape (l, in)
    """
    k = len(activations)
    m = len(activations[0])
    l, in_dim = activations[0][0].shape

    total = torch.zeros(l, in_dim, device=activations[0][0].device)
    for task_acts in activations:       # over k
        for sample in task_acts:        # over m
            total += sample.norm(p=norm, dim=0)  # norm over batch/sample → shape: (l, in)

    return total / (k * m)


@register_importance('wanda')
def compute_wanda_importance(weights, activations, norm=2):
    w_imp = compute_weight_importance(weights)
    a_imp = compute_activation_importance(activations, norm)
    return w_imp * a_imp

def kde_entropy(values, num_points=100):
    from scipy.stats import gaussian_kde
    import numpy as np
    values_np = values.float().cpu().numpy()
    kde = gaussian_kde(values_np)
    xs = np.linspace(values_np.min(), values_np.max(), num_points)
    probs = kde(xs)
    probs /= probs.sum() + 1e-8  # normalize (not strictly necessary)
    entropy = -np.sum(probs * np.log(probs + 1e-8)) * (xs[1] - xs[0]) * 1000
    return torch.tensor(entropy, dtype=values.dtype, device=values.device)

def hist_entropy(values, bins):
    """
    Compute entropy using histogram method.
    
    Args:
        values (Tensor): 1D tensor of values
        bins (int): Number of bins for histogram
    
    Returns:
        float: Entropy value
    """
    hist = torch.histc(values, bins=bins, min=values.min().item(), max=values.max().item())
    probs = hist / (hist.sum() + 1e-8)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
    return entropy

@register_importance('entropy')
def compute_entropy_importance(activations, weights=None, bins=30, gamma_=0.5, lambda_=0.5):
    """
    activations: List[List[Tensor]]  # shape (k tasks, m samples), each: (l, c)
    Returns: Tensor (l, c)
    """

    def compute_entropy_per_channel(act: torch.Tensor):
        """
        act: shape (m, c)
        """
        m, c = act.shape
        entropies = []
        for i in range(c):
            values = act[:, i].abs().cpu()
            entropy = kde_entropy(values)  # or hist_entropy(values, bins)
            entropies.append(entropy)
        return torch.stack(entropies)

    k = len(activations)
    l, c = activations[0][0].shape

    entropy_per_task = []
    for task_acts in activations:
        task_entropy = []
        for layer_idx in range(l):
            # stack m samples for this layer
            act_mat = torch.stack([sample[layer_idx] for sample in task_acts], dim=0)  # (m, c)
            ent = compute_entropy_per_channel(act_mat)
            task_entropy.append(ent)
        entropy_per_task.append(torch.stack(task_entropy))  # (l, c)

    entropy_tensor = torch.stack(entropy_per_task)  # shape (k, l, c)
    mean_entropy = entropy_tensor.mean(dim=0)
    std_entropy = entropy_tensor.std(dim=0, unbiased=False)
    importance = gamma_ * mean_entropy - lambda_ * std_entropy
    return importance

if __name__ == '__main__':
    act = [[torch.randn(28, 8960), torch.randn(28, 8960)]]
    weights = [[torch.randn(28, 1536, 8960), torch.randn(28, 1536, 8960)]]
    multi_act = [[torch.randn(28, 8960), torch.randn(28, 8960)],
                 [torch.randn(28, 8960), torch.randn(28, 8960)],
                 [torch.randn(28, 8960), torch.randn(28, 8960)],]
    importance_e = get_importance('entropy', activations=multi_act)
    importance_a = get_importance('activation', activations=act)
    importance_w = get_importance('weight', weights=weights)
    importance_wanda = get_importance('wanda', activations=act, weights=weights)
    ic(importance_e.shape, importance_a.shape, importance_w.shape, importance_wanda.shape)
    # ic(importance_a.shape)