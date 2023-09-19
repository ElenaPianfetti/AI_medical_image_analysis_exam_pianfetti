# imports
import numpy as np
import pandas as pd
from PIL import Image
import wandb


# built-in imports
import os
import logging
logger = logging.getLogger("ShigellaLog")

class CustomFormatter(logging.Formatter):
    
    def __init__(self, fmt_str, style):
        super().__init__()
        self.style = style
        self.fmt_str = fmt_str
        white = "\x1b[37;20m"
        reset = "\x1b[0m"
        
        self.formats = {logging.DEBUG: white + self.fmt_str + reset, logging.INFO: white + self.fmt_str + reset,
                logging.WARNING: white + self.fmt_str + reset, logging.ERROR: white + self.fmt_str + reset,
                logging.CRITICAL: white + self.fmt_str + reset}

        
    def format(self, record):
        """Format the log"""
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt, style=self.style)
        return formatter.format(record)

def parse_csv(df):
    # the first four rows are the concentrations
    concentrations = df.iloc[0:8, 1:]
    concentrations.index = ['r01', 'r02', 'r03', 'r04', 'r05', 'r06', 'r07', 'r08']
    concentrations.columns = ['c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10', 'c11', 'c12']
    names = concentrations.applymap(lambda x: np.nan if (pd.isna(x) or ';' not in x) else x.split(';')[0])
    concentrations = concentrations.applymap(lambda x: np.nan if (pd.isna(x) or ';' not in x) else float(x.split(';')[1]))
    # other metadata
    experiment_name = df.iloc[10, 0]
    channel_names = df.iloc[13:17, 0].to_list()
    channel_names = [' '.join(name.split(' ')[1:]) for name in channel_names]
    n_wells = df.iloc[18, 0].split(':')[1]
    n_fows = df.iloc[20, 0].split(':')[1]
    n_stacks = df.iloc[22, 0].split(':')[1]
    z_step = df.iloc[24, 0].split(':')[1]

    data = {"names": names,
            "concentrations": concentrations,
            "experiment_name": experiment_name,
            "channel_names": channel_names,
            "n_wells": n_wells,
            "n_fows": n_fows,
            "n_stacks": n_stacks,
            "z_step": z_step}
    return data

def get_labels(df, min_c=0.2, max_c=20, positive_control='0302B17', negative_control='NEGATIVE', unknown=2):
    df['label'] = unknown
    df['label'] = df.apply(lambda x: 0 if x['mab_name'] == negative_control else x['label'], axis=1)
    df['label'] = df.apply(lambda x: 1 if ((min_c <= x['concentration'] <= max_c) and (x['mab_name']==positive_control)) else x['label'], axis=1)
    df = df.drop(df[(df['mab_name'] == positive_control) & (df['label'] == unknown)].index)
    df = df.reset_index(drop=True)
    return df

def create_master(args):
    master_df = pd.DataFrame()

    images_path = os.path.join(args.experiments_path, 'images')
    layouts_path = os.path.join(args.experiments_path, 'layouts')

    plates = sorted(os.listdir(images_path))
    for experiment in plates:
        images = os.listdir(os.path.join(images_path, experiment))
        images = sorted(set([image[:-8] for image in images]))
        rows = [image[0:3] for image in images]
        cols = [image[3:6] for image in images]
        wells = [image[0:6] for image in images]
        layout = pd.read_excel(os.path.join(layouts_path, experiment + '.xlsx'))
        data = parse_csv(layout)
        concentrations = [data['concentrations'].loc[row, col] for row, col in zip(rows, cols)]
        mab_name = [data['names'].loc[row, col] for row, col in zip(rows, cols)]
        experiment_df = pd.DataFrame({'plate': [experiment] * len(images), 'image': images, 'concentration': concentrations, 'mab_name': mab_name, "well": wells})
        master_df = pd.concat([master_df, experiment_df], ignore_index=True)
    
    return master_df

def filter_master(args, df):
    exc_index = []
    logger.info(f'Original master shape: {df.shape}')

    # first filter: remove images with no mab_name
    for i in range(df.shape[0]):
        if pd.isna(df.iloc[i]['mab_name']):
            exc_index.append(i)
    # drop rows with index in exc_index
    df = df.drop(exc_index)
    df = df.reset_index(drop=True)
    exc_index = []
    logger.info(f'After first filter (remove images not treated with mab): {df.shape}')

    # second filter: remove images that have concentration 20 and plate vOPK230320_THP1_Ss951_mCherry_Gent_Arl8-CellMask(ER)
    for i in range(df.shape[0]):
        if (df.iloc[i]['plate'] == 'vOPK230320_THP1_Ss951_mCherry_Gent_Arl8-CellMask(ER)') and (df.iloc[i]['mab_name'] == '0302B17') and (df.iloc[i]['concentration'] == 20):
            exc_index.append(i)
    # drop rows with index in exc_index
    df = df.drop(exc_index)
    df = df.reset_index(drop=True)
    exc_index = []
    logger.info(f'After second filter (remove images with concentration 20 and plate vOPK230320_THP1_Ss951_mCherry_Gent_Arl8-CellMask(ER)): {df.shape}')
    
    # third filter: remove images that are completely black
    for i in range(df.shape[0]):
        im_path = os.path.join(args.experiments_path, 'images', df.iloc[i]['plate'], df.iloc[i]['image'])
        ch1 = Image.open(im_path + '-ch1.png')
        _, max_ch1 = ch1.getextrema()
        if max_ch1 == 0:
            exc_index.append(i)
    df = df.drop(exc_index)
    df = df.reset_index(drop=True)
    exc_index = []
    logger.info(f'After third filter (remove images that are completely black): {df.shape}')

    # fourth filter: remove outliers with no cells
    outliers_230320 = ['r01c03f09', 'r01c04f09', 'r01c06f08', 'r01c06f09',
                       'r02c01f09', 
                       'r04c02f01']
    outliers_230619 = ['r03c04f09']
    outliers_230626 = ['r01c04f09', 'r01c05f09', 'r01c12f10',
                       'r02c02f10', 'r02c12f10',
                       'r03c03f08', 'r03c04f03']
    print('Have to remove ', len(outliers_230320) + len(outliers_230626) + len(outliers_230619), ' outliers')
    for i in range(df.shape[0]):
        if (df.iloc[i]['plate'] == 'vOPK230320_THP1_Ss951_mCherry_Gent_Arl8-CellMask(ER)') and (df.iloc[i]['image'] in outliers_230320):
            exc_index.append(i)
        if (df.iloc[i]['plate'] == 'vOPK230619_THP1_Ss951_mCherry_NoOPS_Arl8-CellMask(ER)') and (df.iloc[i]['image'] in outliers_230619):
            exc_index.append(i)
        if (df.iloc[i]['plate'] == 'vOPK230626_THP1_Ss951_mCherry_NoOPS_Arl8-CellMask(ER)') and (df.iloc[i]['image'] in outliers_230626):
            exc_index.append(i)

    df = df.drop(exc_index)
    df = df.reset_index(drop=True)
    exc_index = []
    logger.info(f'After fourth filter (remove outliers that have no cells): {df.shape}')

    return df

def set_logger(file_name) -> None:
    """Set up file and console handlers for logger"""
    logger = logging.getLogger("ShigellaLog")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(file_name, mode='w')
    fh.setLevel(logging.DEBUG)
	# create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
	# create a formatter and add it to the handlers
    file_format = "{levelname:<10} | {module:<10} {funcName:<15} | line {lineno:>3}: {message}\n"
    console_format = "{message}"
    ch.setFormatter(CustomFormatter(console_format, style='{'))
    fh.setFormatter(logging.Formatter(file_format, style='{'))
	# add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

def wandb_setup(args) -> None:
    wandb.init(project="Fluorescenza", entity="elenapianfetti", config=vars(args))
    wandb.define_metric("epoch")
    wandb.define_metric("validation/loss", step_metric="epoch")
    wandb.define_metric("validation/acc", step_metric="epoch")
    wandb.define_metric("validation/prec", step_metric="epoch")
    wandb.define_metric("validation/f1", step_metric="epoch")
    wandb.define_metric("validation/tp", step_metric="epoch")
    wandb.define_metric("validation/tn", step_metric="epoch")
    wandb.define_metric("validation/fp", step_metric="epoch")
    wandb.define_metric("validation/fn", step_metric="epoch")
    wandb.define_metric("train/loss", step_metric="epoch")
    wandb.define_metric("train/acc", step_metric="epoch")
    wandb.define_metric("train/prec", step_metric="epoch")
    wandb.define_metric("train/rec", step_metric="epoch")
    wandb.define_metric("train/f1", step_metric="epoch")
    wandb.define_metric("train/tp", step_metric="epoch")
    wandb.define_metric("train/tn", step_metric="epoch")
    wandb.define_metric("train/fp", step_metric="epoch")
    wandb.define_metric("train/fn", step_metric="epoch")