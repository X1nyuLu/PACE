NET = {
    "dim": 1024,
    "depth": 4,
    "kernel_size": 9,
    "patch_size": 16,
    "pool_dim": 256
}
STRATEGY = {
    'train': {
        "batch_size": 64,
        "epoch": 200,
        "patience": 200,
        'train_size': 0.8,
        "optmizer": "Adam",
        "Adam_params": {"lr": 1e-4}, 
        # "Adam_params": {"lr": 1e-3}, # for Bacteria
    },
    'tune': {
        "batch_size": 64,
        # "batch_size": 8, # for Bacteria
        "epoch": 200,
        "patience": 50,
        'train_size': 0.8,
        "optmizer": "Adam",
        "Adam_params": {"lr": 1e-5},
    }
}
