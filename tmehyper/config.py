"""Model and training configuration of TMEHyper.  Every value below is the one used for the reported experiments."""
import torch


class Config:
    # ---- data ----
    data_file = "GSE274447.h5ad"
    n_genes = 500                    # gene panel: the 500 most variable genes of the data file (if the file has more)
    max_regulariser_nodes = 50000    # the hyperedge regulariser is built on all cells up to this size, else on the perturbed set only

    # ---- split ----
    split_mode = "random"            # random: knock-out cells split 80/20 at random (seed) | file: train / test indices from split_file
    split_file = None
    train_ratio = 0.8

    # ---- locally paired control (pre-perturbation state) ----
    pair_k = 10                      # the paired control is one of the pair_k nearest same-type control cells, redrawn every epoch

    # ---- cell encoder ----
    cell_embedding_file = None       # frozen foundation-model embedding (<stem>.npy + <stem>.obs_names.txt), e.g. Nicheformer for mouse data
    cell_hidden_dim = 512            # hidden width of the perceptron on a precomputed embedding
    scgpt_model_dir = None           # scGPT checkpoint directory (vocab.json, args.json, best_model.pt) for human data
    scgpt_embed_dim = 512            # read from the checkpoint
    scgpt_n_bins = 51
    scgpt_pad_value = -2
    max_seq_len = 1200
    latent_dim = 512

    # ---- environment description and group hypergraph ----
    env_n_domains = 8
    group_max_sources = 40
    env_embed_dim = 32

    # ---- perturbation channel and response head ----
    perturb_embed_dim = 32
    group_rank = 64
    gate_kl = 0.01                   # penalty on the divergence of the response gate from the uniform gate

    # ---- gene gate ----
    gene_attn_dim = 64
    gene_attn_heads = 4
    gate_l1 = 0.01

    # ---- decoder ----
    bottleneck_dim = 256
    decoder_hidden_dim = 512
    n_decoder_layers = 4
    dropout = 0.1

    # ---- adversary ----
    adv_weight = 0.1
    adv_lambda_max = 1.0

    # ---- hyperedge regularisers on the environment embedding (cohesion within a hyperedge, separation between hyperedges) ----
    regulariser_k = 30
    regulariser_sigma = 5.0
    regulariser_max_edges = 100      # hyperedges sampled per batch
    cohesion_weight = 0.1
    separation_weight = 0.01
    separation_margin = 0.5

    # ---- environment-matched reference ----
    ref_file = None                  # reference profile of every perturbed-set cell (scripts/build_reference.py); its offset is calibrated per gene

    # ---- training ----
    batch_size = 128
    epochs = 100
    lr = 1e-4
    weight_decay = 0.01
    grad_clip = 1.0
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self):
        out = {k: v for k, v in vars(type(self)).items() if not k.startswith("_") and not callable(v)}
        out.update({k: v for k, v in vars(self).items() if not k.startswith("_")})
        return {k: v for k, v in out.items() if isinstance(v, (int, float, str, bool, list, tuple, type(None)))}
