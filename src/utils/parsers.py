# built-in imports
import argparse
import os
import logging
logger = logging.getLogger("ShigellaLog")

def training_parser():
    parser = argparse.ArgumentParser(add_help = False)
    
    parser.add_argument('--img_path', type=str, default='/data1/epianfetti/operaImages/data/images/', help='path to the images')
    parser.add_argument('--model', type=str, help='model architecture', default='adapter4_3', choices=['adapter_4_3', 'encoder_x4', 'encoder_x1'])
    parser.add_argument('--patch_size', type=int, default=512, help='patch size')
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size for inference (lower then training since the images are divided in patches)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    parser.add_argument('--aggregation', type=str, default='majority', help='aggregation method for the patches')
    parser.add_argument('--min_c', type=float, default=0.2, help='minimum concentration for positive class')
    parser.add_argument('--max_c', type=float, default=20, help='maximum concentration for positive class')
    parser.add_argument('--turn_off', type=int, default=None, choices=[1, 2, 3, 4], help='Index of the channel to turn off')
    parser.add_argument('--single_channel', type=int, default=None, help='Index of the channel to keep')
    parser.add_argument('--color_jitter', type=int, choices=[0, 1], help='color jitter', default=0)
    parser.add_argument('--single_plate', type=str, default=None, help='For training on a single plate')

    parser.add_argument('--verbose', type=int, choices=[0, 1], help='verbose mode', default=0)
    parser.add_argument('--dry_run', type=int, choices=[0, 1], default=0, help='dry run')
    parser.add_argument('--df', type=str, help='path to dataframe with info about the plate', required=True)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--virtual_batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--positive_weight', type=float, default=1.5, help='Cross entropy weight for the positive class')
    parser.add_argument('--force_test', type=str, default=None, choices=['low', 'high'], help='Force test on a specific split')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function', choices=['cross_entropy'])
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer', choices=['adam'])
    parser.add_argument('--scheduler', type=str, default='cosineannealing', help='scheduler', choices=['cosineannealing'])
    parser.add_argument('--test_every', type=int, default=1, help='test every n batches')
    parser.add_argument('--split_by', type=str, default=None, choices=['plate', 'well', 'image', None], help='Split by plate, well, or image')
    parser.add_argument('--n_splits', type=int, default=4, help='Number of splits for cross-validation')
    parser.add_argument('--classifier_finetuning', type=int, choices=[0, 1], help='0 for finetuning entire net, 1 for finetuning the classifier', default=0)
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint', default=None)

    # ---- misc parameters -----
    parser.add_argument('--wandb', type=int, choices=[0, 1], help='use wandb for logging', default=1)
    parser.add_argument('--results', type=str, help='path to save the results', required=True)
    parser.add_argument('--debug', type=int, choices=[0, 1], help='debug mode', default=0)

    args = parser.parse_args()

    # checks
    # virtual batch size must be a multiple of batch size
    assert args.virtual_batch_size % args.batch_size == 0, "virtual batch size must be a multiple of batch size"
    # turn off and single channel are mutually exclusive
    assert not (args.turn_off is not None and args.single_channel is not None), "You cannot turn off a channel with the model that keeps only one channel"
    # single_channel can only be used if the model is encoder_x1
    assert not (args.single_channel is not None and args.model != 'encoder_x1'), "You can only keep one channel with the encoder_x1 model"
    # turn_off can only be used if the model is adapter4_3, or encoder_x4
    assert not (args.turn_off is not None and args.model not in ['adapter4_3', 'encoder_x4']), "You can only turn off a channel with the adapter4_3 or encoder_x4 model"
    # patience should be less than epochs
    assert args.patience < args.epochs, "Patience should be less than epochs"

    # if finetuning, checkpoint must be provided, and a single plate must be provided
    if args.classifier_finetuning:
        assert args.checkpoint is not None, "Checkpoint must be provided if finetuning"
        assert args.single_plate is not None, "Single plate must be provided if finetuning"
    
    
    if not args.dry_run:
        # check if the results folder exists
        if not os.path.exists(args.results):
            print(f"Creating folder {args.results}")
            os.makedirs(args.results)
            os.makedirs(os.path.join(args.results, 'splits'))
            os.makedirs(os.path.join(args.results, 'models'))
            os.makedirs(os.path.join(args.results, 'test_res'))
        else:
            if args.results == 'results/tests':
                pass
            else:
                raise ValueError(f"Folder {args.results} already exists you might overwrite something")

    # check if the dataframe file exists
    if not os.path.exists(args.df):
        raise ValueError(f"Dataframe file {args.df} does not exist")

    return args