import torch
from gradSE.model import GradDeReverb

def init_gradSE(checkpointpath=None):
    
    if checkpointpath == None:
        import gradSE.params as params
        model = GradDeReverb(params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
        ckpt = torch.load(f"{params.log_dir}/grad_{params.checkpoint_epoch}.pt")
        model.load_state_dict(ckpt['model'])
    return model
