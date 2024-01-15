train_dirpath_path = '/srv/storage/talc@talc-data.nancy/multispeech/calcul/users/sdowerah/se_sv_voxceleb_traindata'
valid_dirpath_path = '/srv/storage/talc@talc-data.nancy/multispeech/calcul/users/sdowerah/se_sv_voxceleb_valdata'

feattype = 'mel'# stft or mel
mode = 1# 3 modes
n_feats = 40

random_seed = 37

dec_dim = 128
beta_min = 0.005
beta_max = 5.0
pe_scale = 1000

# eval

outdir = './out/'
import os
if os.path.exists(outdir) == False:
    os.mkdir(outdir)

# training parameters
log_dir = '/srv/storage/talc@talc-data.nancy/multispeech/calcul/users/sdowerah/grad_se/model1_mel_exp1' # model dir path
checkpoint_epoch = 141
test_size = 4
n_epochs = 10000
batch_size = 8
learning_rate = 1e-5
seed = 37
save_every = 1
out_size = 512
