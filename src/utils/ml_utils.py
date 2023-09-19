# imports
import torch
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

# built-in imports
import os
import logging
logger = logging.getLogger("ShigellaLog")

def apply_suffixes(lst, suffixes):
    n = len(suffixes)
    result = []
    
    for i, item in enumerate(lst):
        suffix = suffixes[i % n]
        result.append(item + suffix)

    return result

def custom_collate(batch):
    """Custom collate function to handle the list of patches"""
    
    # get the list of patches
    patches = [item['image'] for item in batch]
    n_patches = patches[0].shape[0]
    # get the labels
    labels = [item['label'] for item in batch]

    # get image name
    names = [item['image_name'] for item in batch]

    # stack the patches
    patches = torch.cat(patches, dim=0).reshape(-1, patches[0].shape[-3], patches[0].shape[-2], patches[0].shape[-1])
    # stack the labels and all metadata
    labels = torch.tensor(labels).long().unsqueeze(1).repeat(1, patches.shape[0] // len(labels)).reshape(-1)
    suffixes = [f"_p{patch_number+1:02d}" for patch_number in range(n_patches)]
    names = np.repeat(names, n_patches)
    patch_names = apply_suffixes(names, suffixes)
    # collated_batch = {'image': patches, 'label': labels, 'image_name': names}
    collated_batch = {'image': patches, 'label': labels, 'image_name': names, 'patch_names': patch_names}

    return collated_batch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.min_val_loss = np.Inf
        self.early_stop = False

    def __call__(self, val_loss, model, model_path):
        if ((val_loss+self.min_delta) < self.min_val_loss):
            if self.verbose:
                print(f"Validation loss decreased ({self.min_val_loss:.6f} --> {val_loss:.6f}).  Saving model ...")
            torch.save(model.state_dict(), model_path)
            self.min_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss did not decrease {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def split_by(df, path, by=None, plate_name=None, test_size=0.1, n_splits=4, seed=None, force_test=None):

    if seed is not None:
        np.random.seed(seed)

    if by not in ['plate', 'well', 'image', None]:
        raise ValueError(f"Split by {by} not supported")

    if plate_name:
        df = df[df['plate'] == plate_name]
        if df.shape[0] == 0:
            raise ValueError(f"Plate {plate_name} not found in the dataframe")
        if by == 'plate':
            raise ValueError(f"Split by {by} not supported when plate is specified")
    
    logger.info(f"Number of samples: {df.shape[0]}")

    if not by:
        by = 'well' if plate_name else 'plate'

    logger.debug(f"Split by {by}")

    df['plate_well'] = df['plate'] + df['well']

    # --------forced test--------
    if force_test == 'low':
        logger.debug("Forced test on low concentration")
        positive = df[df['label'] == 1]
        concentrations = positive['concentration'].unique()
        # get the lowest concentration
        min_conc = min(concentrations)
        test_df = df[df['concentration'] == min_conc]
        train_df = df[df['concentration'] != min_conc]
    elif force_test == 'high':
        logger.debug("Forced test on high concentration")
        positive = df[df['label'] == 1]
        concentrations = positive['concentration'].unique()
        # get the highest concentration
        max_conc = max(concentrations)
        test_df = df[df['concentration'] == max_conc]
        train_df = df[df['concentration'] != max_conc]
    else:
        # --------plate split--------
        if by == 'plate':
            n_el_p = sorted(set(df['plate']))
            logger.debug(f"Number of plates: {len(n_el_p)}")
            # there has to be at least one plate in the test set
            min_test_plates = 1
            n_test_plates = max(min_test_plates, round(len(n_el_p) * test_size))
            logger.debug(f"Number of plates in the test set: {n_test_plates}")

            # pick n_test_plates randomly from n_el_p
            test_plates = np.random.choice(n_el_p, n_test_plates, replace=False)

            test_indices = df.index[df['plate'].isin(test_plates)].tolist()
            train_indices = df.index[~df['plate'].isin(test_plates)].tolist()
            
            test_df = df.loc[test_indices]
            train_df = df.loc[train_indices]

        # --------well split--------
        if by == 'well':
            n_el_w = sorted(set(df['plate'] + df['well']))
            # dividing by well we have to check that both classes are present in the test set
            logger.debug(f"Number of wells: {len(n_el_w)}")

            cl_wells = [df['label'].loc[df['plate'] + df['well'] == well].unique() for well in n_el_w]
            for a, b in zip(n_el_w, cl_wells):
                if len(b) != 1:
                    raise ValueError(f"Well {a} has more than one class")
            cl_wells = [b[0] for b in cl_wells]
            # count the number of positive and negative wells
            n_pos_wells = cl_wells.count(1)
            n_neg_wells = cl_wells.count(0)
            minority_wells = min(n_pos_wells, n_neg_wells)

            min_well_class_test = 2
            n_test_wells_class = max(min_well_class_test, round(minority_wells * test_size))

            # elements are not picked from different plates (or at least at random one plate could not be picked in the test)
            # pick n_test_wells randomly from n_el_w (for each class)
            pos_wells = np.random.choice([w for w, c in zip(n_el_w, cl_wells) if c == 1], n_test_wells_class, replace=False)
            neg_wells = np.random.choice([w for w, c in zip(n_el_w, cl_wells) if c == 0], n_test_wells_class, replace=False)
            test_wells = np.concatenate((pos_wells, neg_wells))
            logger.debug(f"Number of wells in the test set: {len(test_wells)}")

            test_indices = df.index[(df['plate'] + df['well']).isin(test_wells)].tolist()
            train_indices = df.index[~(df['plate'] + df['well']).isin(test_wells)].tolist()
            
            test_df = df.loc[test_indices]
            train_df = df.loc[train_indices]

        # --------image split--------
        if by == 'image':
            n_el_r = sorted(set(df['plate'] + df['image']))
            logger.debug(f"Number of images: {len(n_el_r)}")

            cl_images = [df['label'].loc[df['plate'] + df['image'] == image].unique() for image in n_el_r]

            for a, b in zip(n_el_r, cl_images):
                if len(b) != 1:
                    raise ValueError(f"Image {a} has more than one class")
            cl_images = [b[0] for b in cl_images]
                # count the number of positive and negative images
            n_pos_images = cl_images.count(1)
            n_neg_images = cl_images.count(0)
            minority_images = min(n_pos_images, n_neg_images)

            min_images_class_test = 20
            n_test_images_class = max(min_images_class_test, round(minority_images * test_size))

            # elements are not picked from different plates (or at least at random one plate could not be picked in the test)
            # pick n_test_images randomly from n_el_r (for each class)
            pos_images = np.random.choice([w for w, c in zip(n_el_r, cl_images) if c == 1], n_test_images_class, replace=False)
            neg_images = np.random.choice([w for w, c in zip(n_el_r, cl_images) if c == 0], n_test_images_class, replace=False)
            test_images = np.concatenate((pos_images, neg_images))
            
            test_indices = df.index[(df['plate'] + df['image']).isin(test_images)].tolist()
            train_indices = df.index[~(df['plate'] + df['image']).isin(test_images)].tolist()
            
            test_df = df.loc[test_indices]
            train_df = df.loc[train_indices]

    logger.info(f"Number of samples in the test set: {test_df.shape[0]}")
    logger.info(f"Number of samples in the train set: {train_df.shape[0]}")

    train_y = list(train_df['label'])

    # save test
    test_df = test_df.drop(columns=['plate_well'])
    test_df.to_csv(os.path.join(path, 'test.csv'), sep='\t', index=False)

    # stratified cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if by == 'image':
        for n_split, (train_index, val_index) in enumerate(skf.split(train_df, train_y)):
            tr_df = train_df.iloc[train_index]
            val_df = train_df.iloc[val_index]

            tr_df = tr_df.drop(columns=['plate_well'])
            val_df = val_df.drop(columns=['plate_well'])
            tr_df.to_csv(os.path.join(path, f'train_{n_split+1}.csv'), sep='\t', index=False)
            val_df.to_csv(os.path.join(path, f'val_{n_split+1}.csv'), sep='\t', index=False)
    else:
        unique_wells = train_df['plate_well'].unique()
        wells_class = [train_df['label'].loc[train_df['plate_well'] == well].unique() for well in unique_wells]
        for n_split, (train_index, val_index) in enumerate(skf.split(unique_wells, wells_class)):
            train_wells = unique_wells[train_index]
            val_wells = unique_wells[val_index]
            tr_df = train_df.loc[train_df['plate_well'].isin(train_wells)]
            val_df = train_df.loc[train_df['plate_well'].isin(val_wells)]
            tr_df = tr_df.drop(columns=['plate_well'])
            val_df = val_df.drop(columns=['plate_well'])
            tr_df.to_csv(os.path.join(path, f'train_{n_split+1}.csv'), sep='\t', index=False)
            val_df.to_csv(os.path.join(path, f'val_{n_split+1}.csv'), sep='\t', index=False)
        

def train(model, dataloader, loss_fn, optimizer, device, args):
    model.train()
    epoch_loss = 0
    virtual_iters = args.virtual_batch_size // args.batch_size
    optimizer.zero_grad()
    with  tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
        for i, batch in pbar:
            # get the inputs; data is a list of [inputs, labels]
            if args.single_channel:
                inputs = batch['image'][:, args.single_channel-1].to(device)
            else:
                inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            # zero the parameter gradients
            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            if (i+1) % virtual_iters == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_description(f"Loss: {loss.item():.3f}")
            epoch_loss += loss.item()
            if args.wandb:
                wandb.log({'train/current_train_loss': loss.item()})
    epoch_loss /= len(dataloader)
    return epoch_loss

@torch.no_grad()
def test(model, dataloader, device, args, loss_fn=None, train=False):
    """Function for testing the model in evaluation mode"""
    model.eval()
    if loss_fn:
        test_loss = 0
    else:
        test_loss = np.nan
    n_patches = (2048 // args.patch_size) ** 2
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    image_df = pd.DataFrame(columns=['image_name', 'label', 'prediction'])
    patch_df = pd.DataFrame(columns=['image_name_patch', 'label', 'prediction'])


    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # get the inputs; data is a list of [inputs, labels]
        if args.single_channel:
            inputs = batch['image'][:, args.single_channel-1].to(device)
        else:
            inputs = batch['image'].to(device)
        labels = batch['label'].to(device)
        names = batch['image_name']
        if not train:
            patch_names = batch['patch_names']
        outputs = model(inputs)
        if loss_fn:
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
        # get the predictions
        _, preds = torch.max(outputs, 1)

        if not train:
            patch_df = pd.concat([patch_df, pd.DataFrame({'image_name_patch': patch_names, 'label': labels.cpu().numpy(), 'prediction': preds.cpu().numpy()})])
            # se è stato fatto il collate allora labels è un tensore di dimensione batch_size * n_patches
            # le predizioni però le voglio image-wise, non patch-wise
            labels = torch.reshape(labels, (-1, n_patches))
            preds = torch.reshape(preds, (-1, n_patches))
            # for the labels, take first element of each row (all patches are the same)
            labels = labels[:, 0]
            # for the names, take first element of each row (all patches are the same)
            names = names[::n_patches]
            if args.aggregation == None:
                pass
            elif args.aggregation == 'majority':
                # for the predictions, take the most frequent element of each row
                preds = torch.mode(preds, dim=1)[0]
            elif args.aggregation == 'positive':
                # positive if at least one patch is positive
                preds = torch.any(preds == 1, dim=1).long()
            else:
                raise NotImplementedError(f"Aggregation method {args.aggregation} not implemented")
            
        if not train:
            # concatenate the dataframe
            image_df = pd.concat([image_df, pd.DataFrame({'image_name': names, 'label': labels.cpu().numpy(), 'prediction': preds.cpu().numpy()})])
        
        # get the true positives, true negatives, false positives and false negatives
        tp += torch.sum((preds == 1) & (labels == 1)).item()
        tn += torch.sum((preds == 0) & (labels == 0)).item()
        fp += torch.sum((preds == 1) & (labels == 0)).item()
        fn += torch.sum((preds == 0) & (labels == 1)).item()


    test_loss /= len(dataloader)
    logger.debug(f"tp: {tp}\ttn: {tn}\tfp: {fp}\tfn: {fn}")
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = round(accuracy, 3)
    precision = round(precision, 3)
    recall = round(recall, 3)
    f1 = round(f1, 3)
    return test_loss, accuracy, precision, recall, f1, (tp, tn, fp, fn), image_df, patch_df
