import argparse

UNK_IDX = 0
UNK_WORD = "UUUNKKK"
EVAL_YEAR = "2017"


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_base_parser():
    parser = argparse.ArgumentParser(
        description='Paraphrase using PyTorch')
    parser.register('type', 'bool', str2bool)

    basic_group = parser.add_argument_group('basics')
    # Basics
    basic_group.add_argument('--debug', type="bool", default=False,
                             help='activation of debug mode (default: False)')
    basic_group.add_argument('--save_prefix', type=str, default="experiments",
                             help='saving path prefix')
    basic_group.add_argument('--checkpoint_dir', type=str, default="CheckPoint")
    basic_group.add_argument('--checkpointD_dir', type=str, default="CheckPoint")
    basic_group.add_argument('--checkpointC_dir', type=str, default="CheckPoint")
    basic_group.add_argument('--load_model_from', type=str, default="CheckPoint/",
                             help='path of trained model to eval')  # is_eval
    basic_group.add_argument('--load_model_from1', type=str, default="CheckPoint/",
                             help='path of trained model to eval')  # is_eval
    basic_group.add_argument('--pretrain_modelD', type=str, default="CheckPoint/",
                             help='dir of pretrained disentangle model')  # use_pretrain
    basic_group.add_argument('--pretrain_modelC', type=str, default="CheckPoint/",
                             help='dir of pretrained attack model')  # use_pretrain
    basic_group.add_argument('--continue_from', type=str, default="CheckPoint/",
                             help='dir of model to continue')  # continue_train

    # added
    basic_group.add_argument('--root_dir', type=str, default="")
    basic_group.add_argument('--raw_feature1', type=str, default="")
    basic_group.add_argument('--raw_feature2', type=str, default="")
    basic_group.add_argument('--pair_vector_path', type=str, default="")
    basic_group.add_argument('--embedding_file_path', type=str, default="")
    basic_group.add_argument('--file_name', type=str, default="")
    basic_group.add_argument('--disentangled_vector_path', type=str, default="")
    basic_group.add_argument('--fileTotrain', type=str, default="")
    basic_group.add_argument('--fileTotest', type=str, default="")
    basic_group.add_argument('--pair_size', type=int, default=0)
    basic_group.add_argument('--x_dim1', type=int, default=100)
    basic_group.add_argument('--x_dim2', type=int, default=50)
    basic_group.add_argument('--x_dim3', type=int, default=50)
    basic_group.add_argument('--alpha', type=float, default=0.1)
    basic_group.add_argument('--beta', type=float, default=0.0)
    basic_group.add_argument('--momentum', type=float, default=0.0)


    data_group = parser.add_argument_group('data')
    # Data file
    data_group.add_argument('--train_file', type=str, default=None,
                            help='training file')
    data_group.add_argument('--eval_file', type=str, default=None,
                            help='evaluation file')
    data_group.add_argument('--vocab_file', type=str, default=None,
                            help='vocabulary file')
    data_group.add_argument('--embed_file', type=str, default=None,
                            help='pretrained embedding file')

    config_group = parser.add_argument_group('model_configs')
    config_group.add_argument('-m', '--margin',
                              dest='m',
                              type=float,
                              default=0.4,
                              help='margin for the training loss')
    # config_group.add_argument('-m1', '--margin1',
    #                           dest='m',
    #                           type=float,
    #                           default=,
    #                           help='margin for the contrastive loss')
    config_group.add_argument('-lr', '--learning_rate',
                              dest='lr',
                              type=float,
                              default=1e-3,
                              help='learning rate')
    config_group.add_argument('-pratio', '--ploss_ratio',
                              dest='pratio',
                              type=float,
                              default=1.0,
                              help='ratio of position loss')
    config_group.add_argument('-lratio', '--logloss_ratio',
                              dest='lratio',
                              type=float,
                              default=1.0,
                              help='ratio of reconstruction log loss')
    config_group.add_argument('-dratio', '--disc_ratio',
                              dest='dratio',
                              type=float,
                              default=1.0,
                              help='ratio of discriminative loss')
    config_group.add_argument('-plratio', '--para_logloss_ratio',
                              dest='plratio',
                              type=float,
                              default=1.0,
                              help='ratio of paraphrase log loss')
    config_group.add_argument('--eps',
                              type=float,
                              default=1e-4,
                              help='for avoiding numerical issues')
    config_group.add_argument('-edim', '--embed_dim',
                              dest='edim',
                              type=int, default=50,
                              help='size of embedding')
    config_group.add_argument('-dp', '--dropout',
                              dest='dp',
                              type=float, default=0.0,
                              help='dropout probability')
    config_group.add_argument('-gclip', '--grad_clip',
                              dest='gclip',
                              type=float, default=None,
                              help='gradient clipping threshold')
    # recurrent neural network detail
    config_group.add_argument('-ensize', '--encoder_size',
                              dest='ensize',
                              type=int, default=50,
                              help='encoder hidden size')
    config_group.add_argument('-desize', '--decoder_size',
                              dest='desize',
                              type=int, default=50,
                              help='decoder hidden size')
    config_group.add_argument('--ysize',
                              dest='ysize',
                              type=int, default=50,
                              help='size of vMF')
    config_group.add_argument('--zsize',
                              dest='zsize',
                              type=int, default=50,
                              help='size of Gaussian')

    # feedforward neural network
    config_group.add_argument('-mhsize', '--mlp_hidden_size',
                              dest='mhsize',
                              type=int, default=100,
                              help='size of hidden size')
    config_group.add_argument('-mlplayer', '--mlp_n_layer',
                              dest='mlplayer',
                              type=int, default=1,
                              help='number of layer')
    config_group.add_argument('-zmlplayer', '--zmlp_n_layer',
                              dest='zmlplayer',
                              type=int, default=1,
                              help='number of layer')
    config_group.add_argument('-ymlplayer', '--ymlp_n_layer',
                              dest='ymlplayer',
                              type=int, default=1,
                              help='number of layer')

    # optimization
    config_group.add_argument('-mb', '--mega_batch',
                              dest='mb',
                              type=int, default=1,
                              help='size of mega batching')
    config_group.add_argument('-ps', '--p_scramble',
                              dest='ps',
                              type=float, default=0.,
                              help='probability of scrambling')
    config_group.add_argument('--l2', type=float, default=0.,
                              help='l2 regularization')
    config_group.add_argument('-vmkl', '--max_vmf_kl_temp',
                              dest='vmkl', type=float, default=1e-3,
                              help='temperature of kl divergence')
    config_group.add_argument('-gmkl', '--max_gauss_kl_temp',
                              dest='gmkl', type=float, default=1e-4,
                              help='temperature of kl divergence')

    setup_group = parser.add_argument_group('train_setup')
    # train detail
    setup_group.add_argument('--save_dir', type=str, default=None,
                             help='model save path')
    setup_group.add_argument('--saveC_dir', type=str, default=None,
                             help='model save path')
    setup_group.add_argument('--saveD_dir', type=str, default=None,
                             help='model save path')
    basic_group.add_argument('--embed_type',
                             type=str, default="paragram",
                             choices=['paragram', 'glove'],
                             help='types of embedding: paragram, glove')
    basic_group.add_argument('--yencoder_type',
                             type=str, default="word_avg",
                             help='types of encoder for y variable')
    basic_group.add_argument('--zencoder_type',
                             type=str, default="word_avg",
                             help='types of encoder for z encoder')
    basic_group.add_argument('--decoder_type',
                             type=str, default="bag_of_words",
                             help='types of decoder')
    setup_group.add_argument('--n_epoch', type=int, default=5,
                             help='number of epochs')
    setup_group.add_argument('--D_epoch', type=int, default=5,
                             help='number of epochs')
    setup_group.add_argument('--C_epoch', type=int, default=5,
                             help='number of epochs')
    setup_group.add_argument('--batch_size', type=int, default=20,
                             help='batch size')
    setup_group.add_argument('--opt', type=str, default='adam',
                             help='types of optimizer')
    setup_group.add_argument('--pre_train_emb', type="bool", default=False,
                             help='whether to use pretrain embedding')
    setup_group.add_argument('--vocab_size', type=int, default=50000,
                             help='size of vocabulary')

    misc_group = parser.add_argument_group('misc')
    # misc
    misc_group.add_argument('--print_every', type=int, default=10,
                            help='print training details after \
                            this number of iterations')
    misc_group.add_argument('--eval_every', type=int, default=100,
                            help='evaluate model after \
                            this number of iterations')
    misc_group.add_argument('--summarize', type="bool", default=False,
                            help='whether to summarize training stats\
                            (default: False)')
    misc_group.add_argument('--is_eval', type="bool", default=False,
                            help='test or train')
    misc_group.add_argument('--continue_train', type="bool", default=False,
                            help='whether resume train')
    misc_group.add_argument('--use_pretrainD', type="bool", default=False,
                            help='whether use pretrain model')
    misc_group.add_argument('--use_pretrainC', type="bool", default=False,
                            help='whether use pretrain model')
    return parser
