test_filelist_path = 'noisy_400_5_list.txt'
test_dirpath = '/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/sdowerah/interspeech_2021_v1/robovox_400_5/Noisy'

feattype = 'mel'
mode = 1
n_feats = 40



dec_dim = 128
beta_min = 0.005
beta_max = 5.0
pe_scale = 1000

outdir = './out/'
import os
if os.path.exists(outdir) == False:
    os.mkdir(outdir)

log_dir = '/srv/storage/talc@talc-data.nancy/multispeech/calcul/users/sdowerah/grad_se/model1_mel' # model dir path
checkpoint_epoch = 100
