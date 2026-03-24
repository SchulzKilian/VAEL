"""
VAEL experiment(s) configuration
"""
mnist_vael = dict(
    device = 'cuda:0',
    n_digits = 10,
    # Final matched-pair comparison: flow_w=0 vs flow_w=2 (best from ablation).
    # Both conditions: KL annealing over 20 epochs, kl_w=1e-3, everything else identical.
    # Only difference: flow presence and warmup. 10 seeds each for error bars.
    # Combine with 3 existing seeds per condition from prior experiments = 13 total each.
    experiment_name='vael_2digitMNIST_final_comparison',
    dataset_dimensions = {'train': 42000,
                          'val': 12000,
                          'test': 6000},
    batch_size = {'train': 30,
                  'val': 120,
                  'test': 60},
    exp_config = {'task': ['base'],
                  'tag': ['final'],
                  'rec_loss': ['LAPLACE'],
                  'max_epoch': [50],
                  'n_exp': [10],
                  'latent_dim_sub': [8],
                  'latent_dim_sym': [15],
                  'learning_rate': [1e-4],
                  'dropout': [0.5],
                  'dropout_ENC': [0.5],
                  'dropout_DEC': [0.5],
                  'recon_w': [1e-1],
                  'kl_w': [1e-3],
                  'query_w': [1.],
                  'sup_w': [0.],
                  'flow_w': [2],  # best flow weight; fixed across both conditions
                  'no_symbolic': [False, True],  # VAEL+flow vs pure VAE+flow
                  'query': [True]},
    early_stopping_info = {
        'patience': 20,
        'delta': 1e-8})

mario_vael = dict(
    device='cuda:0',
    n_facts=18,
    experiment_name='vael_Mario',
    lagrangian=False,
    batch_size={'train': 32,
                'val': 32,
                'test': 32},
    exp_config={'task': ['base'],
                'tag': ['base'],
                'rec_loss': ['LAPLACE'],
                'max_epoch': [200],
                'n_exp': [1],
                'latent_dim_sub': [30],
                'latent_dim_sym': [18],
                'hidden_chs_MLP': [32],
                'hidden_chs_ENC': [64],
                'hidden_chs_DEC': [64],
                'learning_rate': [1e-4],
                'dropout': [0.1],
                'dropout_ENC': [0.0],
                'dropout_DEC': [0.0],
                'recon_w': [10.],
                'kl_w': [10.],
                'query_w': [1e4],
                'sup_w': [0.],
                'query': [True]},
    early_stopping_info={
        'patience': 20,
        'delta': 1e-8,
        'loss': 'true_elbo'},)
