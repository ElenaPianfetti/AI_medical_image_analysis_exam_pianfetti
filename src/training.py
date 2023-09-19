# imports
import pandas as pd
import wandb
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

# module imports
from model.dataset import Dataset
from model.model import Adapter4_3, Encoder_x1
from model.custom_transforms import TurnOffTransform, MockTransform, SplitPatchTransform
from utils.ml_utils import EarlyStopping, custom_collate, split_by, train, test
from utils.parsers import training_parser
from utils.utils import get_labels, wandb_setup, set_logger

# built-in imports
import os
import logging
logger = logging.getLogger("ShigellaLog")

def main(args):

    # get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[2315.44, 125.87, 13.62, 744.17], std=[2133.23, 62.45, 30.58, 1370.30]),
        transforms.Resize(2048, antialias=False),    # resize so that it is a multiple of patch_size
        TurnOffTransform(args.turn_off) if args.turn_off else MockTransform(),
        # data augmentation
        transforms.RandomCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),transforms.Resize(256, antialias=False)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[2315.44, 125.87, 13.62, 744.17], std=[2133.23, 62.45, 30.58, 1370.30]),
        transforms.Resize(2048, antialias=False),    # resize so that it is a multiple of patch_size
        TurnOffTransform(args.turn_off) if args.turn_off else MockTransform(), 
        SplitPatchTransform(args.patch_size),
        transforms.Resize(256, antialias=False)
    ])

    df = pd.read_csv(args.df, sep='\t')
    labeled_df = get_labels(df, min_c=args.min_c, max_c=args.max_c)
    labeled_df.to_csv('data/test_labels.csv', sep='\t', index=False)

    # split the dataset
    split_by(labeled_df, os.path.join(args.results, 'splits'), by=args.split_by, plate_name=args.single_plate, seed=args.seed, n_splits=args.n_splits, force_test=args.force_test)

    test_df = pd.read_csv(os.path.join(args.results, 'splits', 'test.csv'), sep='\t')
    
    test_dataset = Dataset(df=test_df, transforms=test_transform, path=args.img_path)
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=4, shuffle=False, collate_fn=custom_collate)


    for split in range(args.n_splits):

        if args.wandb:
            wandb_setup(args)

        logger.info(f"Fold {split+1}")
        train_df = pd.read_csv(os.path.join(args.results, 'splits', f'train_{split+1}.csv'), sep='\t')
        val_df = pd.read_csv(os.path.join(args.results, 'splits', f'val_{split+1}.csv'), sep='\t')
        
        train_dataset = Dataset(df=train_df, transforms=train_transform, path=args.img_path)
        val_dataset = Dataset(df=val_df, transforms=test_transform, path=args.img_path)
        logger.info(f"Dataset split into {len(train_dataset)} train samples and {len(val_dataset)} validation samples")
        if not args.debug:
            train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
            val_dataloader = data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=16, collate_fn=custom_collate)
        else:
            train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
            val_dataloader = data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=custom_collate)
    

        # ------------------------------------------------------
        # create model
        if args.model == 'adapter4_3':
            model = Adapter4_3(4, 2).to(device)
        elif args.model == 'encoder_x1':
            model = Encoder_x1(2).to(device)

        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint))
            logger.info(f"Loaded checkpoint {args.checkpoint}")
            if args.classifier_finetuning:
                for param in model.encoder.parameters():
                    param.requires_grad = False
                logger.info("Frozen encoder parameters")
        
        # create optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

        # create loss function
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, args.positive_weight]).float().to(device))

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        for epoch in range(args.epochs):
            logger.info(f"Epoch:\t{epoch+1}")
            train_loss = train(model, train_dataloader, loss_fn, optimizer, device, args)
            scheduler.step()
            metrics = {'train/loss': train_loss}
            if epoch % args.test_every == 0:

                val_loss, acc, prec, rec, f1, cmatrix, _, _ = test(model, val_dataloader, device, args, loss_fn=loss_fn)
                metrics['validation/loss'] = val_loss
                metrics['validation/acc'] = acc
                metrics['validation/prec'] = prec
                metrics['validation/rec'] = rec
                metrics['validation/f1'] = f1
                metrics['validation/tp'] = cmatrix[0]
                metrics['validation/tn'] = cmatrix[1]
                metrics['validation/fp'] = cmatrix[2]
                metrics['validation/fn'] = cmatrix[3]

                train_loss, train_acc, train_prec, train_rec, train_f1, train_cmat, _, _ = test(model, train_dataloader, device, args, loss_fn=loss_fn, train=True)
                metrics['train/acc'] = train_acc
                metrics['train/prec'] = train_prec
                metrics['train/rec'] = train_rec
                metrics['train/f1'] = train_f1
                metrics['train/tp'] = train_cmat[0]
                metrics['train/tn'] = train_cmat[1]
                metrics['train/fp'] = train_cmat[2]
                metrics['train/fn'] = train_cmat[3]

                if epoch == 0:
                    torch.save(model.state_dict(), f"{args.results}/models/fold{split+1}.pt")

                metrics['epoch'] = epoch

                early_stopping(val_loss, model, f"{args.results}/models/fold{split+1}.pt")
                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    if args.wandb:
                        wandb.log(metrics)
                    break
            
            logger.debug(metrics)
            if args.wandb:
                wandb.log(metrics)
        
        # load best model and test on test set
        logger.info(f"Loading best model for fold {split+1}")
        model.load_state_dict(torch.load(f"{args.results}/models/fold{split+1}.pt"))
        test_loss, acc, prec, rec, f1, cmatrix, df, patch_df = test(model, test_dataloader, device, args, loss_fn=loss_fn)
        df.to_csv(f"{args.results}/test_res/fold_{split+1}.csv", index=False, sep='\t')
        patch_df.to_csv(f"{args.results}/test_res/fold_{split+1}_patch.csv", index=False, sep='\t')
        with open(f"{args.results}/test_res/fold_{split+1}_metrics.csv", 'w') as f:
            f.write('----------test results----------\n')
            f.write(f"Accuracy:\t\t\t{acc*100:.2f}%\n")
            f.write(f"Precision:\t\t\t{prec*100:.2f}%\n")
            f.write(f"Recall:\t\t\t\t{rec*100:.2f}%\n")
            f.write(f"F1:\t\t\t\t\t{f1*100:.2f}%\n\n")
            f.write('\n--------Confusion Matrix--------\n')
            f.write(F"\t\t\t\t  Predicted\n")
            f.write(f"\t\t\tPositive\tNegative\n")
            f.write(f"Positive\t{cmatrix[0]}\t\t\t{cmatrix[3]}\n")
            f.write(f"Negative\t{cmatrix[2]}\t\t\t{cmatrix[1]}\n")
        test_metrics = {'test/loss': test_loss, 'test/acc': acc, 'test/prec': prec, 'test/rec': rec, 'test/f1': f1}
        test_metrics['test/tp'] = cmatrix[0]
        test_metrics['test/tn'] = cmatrix[1]
        test_metrics['test/fp'] = cmatrix[2]
        test_metrics['test/fn'] = cmatrix[3]

        if args.wandb:
            wandb.log(test_metrics)
        logger.info(f"\n----------test results------------\n{test_metrics}\n")

        if args.wandb:
            wandb.finish()

    

if __name__ == "__main__":
    args = training_parser()
    if args.verbose or args.dry_run:
        print('\nStarting with arguments:')
        # get longest argument name
        max_len = max([len(arg) for arg in vars(args)])
        for arg in vars(args):
            print(f"{arg:{max_len}} : {getattr(args, arg)}")
        print('\n')
    if not args.dry_run:
        file_name = f'{args.results}/training.log'
        set_logger(file_name)
        main(args)