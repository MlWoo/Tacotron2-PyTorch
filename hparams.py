import tensorflow as tf
import numpy as np


# Default hyperparameters
hparams = tf.contrib.training.HParams(
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    # text, you may want to use "basic_cleaners" or "transliteration_cleaners".
    cleaners='chinese_cleaners',
    builder='Tacotron2',

    # Basic config
    seed=1234,
    cudnn_deterministic=False,
    cudnn_benchmark=False,

    #Hardware setup (TODO: multi-GPU parallel tacotron training)
    device=0,
    world_size=1,
    dist_backend="nccl",
    dist_url="tcp://localhost:54321",

    ####################################################################################################################

    #Audio
    num_mels = 80, #Number of mel-spectrogram channels and local conditioning dimensionality
    num_freq = 513, # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
    rescale = True, #Whether to rescale audio prior to preprocessing
    rescaling_max = 0.999, #Rescaling value
    trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    clip_mels_length = True, #For cases of OOM (Not really recommended, working on a workaround)
    max_mel_frames = 900,  #Only relevant when clip_mels_length = True
    padding_mels = -0.1,

    # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
    # It's preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
    # Does not work if n_ffit is not multiple of hop_size!!
    use_lws=True,
    silence_threshold=2, #silence threshold used for sound trimming for wavenet preprocessing

    #Mel spectrogram
    n_fft = 1024, #Extra window size is filled with 0 paddings to match this parameter
    hop_size = 256, #For 22050Hz, 275 ~= 12.5 ms
    win_size = None, #For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft)
    sample_rate = 22050, #22050 Hz (corresponding to ljspeech dataset)
    frame_shift_ms = None,

    #M-AILABS (and other datasets) trim params
    trim_fft_size = 512,
    trim_hop_size = 128,
    trim_top_db = 60,

    #Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization = True,
    allow_clipping_in_normalization = True, #Only relevant if mel_normalization = True
    symmetric_mels = True, #Whether to scale the data to be symmetric around 0
    max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] 

    #Limits
    min_level_db = -100,
    ref_level_db = 20,
    fmin = 125, #Set this to 75 if your speaker is male! if female, 125 should help taking off noise. (To test depending on dataset)
    fmax = 7600,
    #Contribution by @begeekmyfriend
    #Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
    preemphasize = True, #whether to apply filter                                                                                                                                                             
    preemphasis = 0.97, #filter coefficient.

    ##############################################################################################################################

    #Tacotron
    frames_per_step = 2, #number of frames to generate at each decoding step (speeds up computation and allows for higher batch size)
    stop_at_any = True, #Determines whether the decoder should stop when predicting <stop> to any frame or to all of them
    max_decoder_steps = 800,

	dim_embedding = 512, #dimension of embedding space

    enc_conv_num_layers = 3, #number of encoder convolutional layers
	enc_kernel_size = 5, #size of encoder convolution filters for each layer
    enc_conv_channels = 512, #number of encoder convolutions filters for each layer
	dim_encoder = 256, #number of lstm units for each direction (forward and backward)

    smoothing = False, #Whether to smooth the attention normalization function 
    dim_attention = 128, #dimension of attention space
    attention_filters = 32, #number of attention convolution filters
    attention_kernel = (31, ), #kernel size of attention convolution
    cumulative_weights = True, #Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)

    dim_prenet = 256,
    num_layers = 2, #number of decoder lstm layers
    num_location_features=32,
    gate_threshold=0.5,
    dec_num_filters=512,
    dec_kernel_size=5,
    dim_decoder = 1024, #number of decoder lstm units on each layer
    max_iters = 2500, #Max decoder steps during inference (Just for safety from infinite loop cases)

    postnet_num_layers = 5, #number of postnet convolutional layers
    postnet_kernel_size = (5, ), #size of postnet convolution filters for each layer
    postnet_channels = 512, #number of postnet convolution filters for each layer

    mask_encoder = True, #whether to mask encoder padding while computing attention
    mask_decoder = True, #Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not be weighted, else recommended pos_weight = 20)

    cross_entropy_pos_weight = 20, #Use class weights to reduce the stop token classes imbalance (by adding more penalty on False Negatives (FN)) (1 = disabled)
    ###########################################################################################################################################


    #Tacotron Training
    dynamical_batch_size=False,
    pin_memeory=False, #Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)
    save_optimizer_state = True,

    batch_size = 32, #number of training samples on each training steps
    batch_size_level=4,
    batch_group=32,
    batch_group_size=1536,
    permutate=True,
    clip_thresh = 0.1,
    print_freq=10,

    nepochs=200,
    weight_decay = 1e-7,  #regularization weight (for L2 regularization)
    scale_regularization = True, #Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is high and biasing the model)

    test_size = None, #% of data to keep as test data, if None, tacotron_test_batches must be not None
    test_batches = 24, #number of test batches (For Ljspeech: 10% ~= 41 batches of 32 samples)
    data_random_state=1234, #random state for train test split repeatability

    decay_learning_rate = True, #boolean, determines if the learning rate will follow an exponential decay
    start_decay = 50000, #Step at which learning decay starts
    decay_step = 40000, #Determines the learning rate decay slope (UNDER TEST)
    decay_rate = 0.2, #learning rate decay rate (UNDER TEST)
    init_learning_rate = 1e-3, #starting learning rate
    final_learning_rate = 1e-5, #minimal learning rate

    adam_beta1 = 0.9, #AdamOptimizer beta1 parameter
    adam_beta2 = 0.999, #AdamOptimizer beta2 parameter
    adam_epsilon = 1e-6, #AdamOptimizer beta3 parameter

    zoneout_rate = 0.1, #zoneout rate for all LSTM cells in the network
    dropout_rate = 0.5, #dropout rate for all convolutional layers + prenet

    natural_eval = False, #Whether to use 100% natural eval (to evaluate Curriculum Learning performance) or with same teacher-forcing ratio as in training (just for overfit)

    #Decoder RNN learning can take be done in one of two ways:
    #    Teacher Forcing: vanilla teacher forcing (usually with ratio = 1). mode='constant'
    #    Curriculum Learning Scheme: From Teacher-Forcing to sampling from previous outputs is function of global step. (teacher forcing ratio decay) mode='scheduled'
    #The second approach is inspired by:
    #Bengio et al. 2015: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
    #Can be found under: https://arxiv.org/pdf/1506.03099.pdf
    teacher_forcing_mode = 'constant', #Can be ('constant' or 'scheduled'). 'scheduled' mode applies a cosine teacher forcing ratio decay. (Preference: scheduled)
    teacher_forcing_ratio = 1., #Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs, Only relevant if mode='constant'
    teacher_forcing_init_ratio = 1., #initial teacher forcing ratio. Relevant if mode='scheduled'
    teacher_forcing_final_ratio = 0., #final teacher forcing ratio. Relevant if mode='scheduled'
    teacher_forcing_start_decay = 10000, #starting point of teacher forcing ratio decay. Relevant if mode='scheduled'
    teacher_forcing_decay_steps = 280000, #Determines the teacher forcing ratio decay slope. Relevant if mode='scheduled'
    teacher_forcing_decay_alpha = 0., #teacher forcing ratio decay rate. Relevant if mode='scheduled'
    ###########################################################################################################################################

    #Eval sentences (if no eval file was specified, these sentences are used for eval)
    sentences=[
    # From July 8, 2017 New York Times:
    'Scientists at the CERN laboratory say they have discovered a new particle.',
    'There\'s a way to measure the acute emotional intelligence that has never gone out of style.',
    'President Trump met with other leaders at the Group of 20 conference.',
    'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
    ]

    )

def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
    return 'Hyperparameters:\n' + '\n'.join(hp)
