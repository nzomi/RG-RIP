FFN_MAPPING_BY_MODEL_TYPE = {
    "internlm2": {
        "prefix": "feed_forward",
        "assign": lambda layer, gate, down, up: (
            setattr(layer.feed_forward, "w1", gate),
            setattr(layer.feed_forward, "w2", down),
            setattr(layer.feed_forward, "w3", up),
        ),
        "name_map": {"w1": "gate", "w2": "down", "w3": "up"},
    },
    "qwen2": {
        "prefix": "mlp",
        "assign": lambda layer, gate, down, up: (
            setattr(layer.mlp, "gate_proj", gate),
            setattr(layer.mlp, "down_proj", down),
            setattr(layer.mlp, "up_proj", up),
        ),
        "name_map": {"gate_proj": "gate", "down_proj": "down", "up_proj": "up"},
    },
}
