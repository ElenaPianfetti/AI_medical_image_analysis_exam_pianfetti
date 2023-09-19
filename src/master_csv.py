# built-in imports
import argparse
import os
import datetime
import logging
logger = logging.getLogger("ShigellaLog")

# module imports
from utils.utils import create_master, filter_master, set_logger


def main():
    # the log file will be named master + date + time + .log
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    time = datetime.datetime.now().strftime("%H-%M-%S")
    file_name = f'master_{date}_{time}.log'
    set_logger(file_name)
    parser = argparse.ArgumentParser(description='Create master files for entire dataset and for each plate')
    parser.add_argument('--experiments_path', default='/data1/epianfetti/operaImages/data', type=str, help='path to experiments')
    args = parser.parse_args()
    
    m_df = create_master(args)
    m_df = filter_master(args, m_df)
    # divide master df into a dataframe with positive and negative controls, and a dataframe with unknown mabs
    controls = ['NEGATIVE', '0302B17']
    controls_df = m_df[m_df['mab_name'].isin(controls)]
    unknown_df = m_df[~m_df['mab_name'].isin(controls)]
    m_df.to_csv(os.path.join(args.experiments_path, 'master.csv'), index=False, sep='\t')
    controls_df.to_csv(os.path.join(args.experiments_path, 'controls.csv'), index=False, sep='\t')
    unknown_df.to_csv(os.path.join(args.experiments_path, 'unknown.csv'), index=False, sep='\t')
    

    plates = sorted(set(m_df['plate']))

    for plate in plates:
        p_df = m_df[m_df['plate'] == plate]
        p_df.to_csv(os.path.join(args.experiments_path, 'dataset_df', plate + '.csv'), index=False, sep='\t')
        p_controls_df = controls_df[controls_df['plate'] == plate]
        p_controls_df.to_csv(os.path.join(args.experiments_path, 'dataset_df', plate + '_controls.csv'), index=False, sep='\t')
        p_unknown_df = unknown_df[unknown_df['plate'] == plate]
        p_unknown_df.to_csv(os.path.join(args.experiments_path, 'dataset_df', plate + '_unknown.csv'), index=False, sep='\t')
        logger.info(f'Plate {plate} has {p_df.shape[0]} images\n{p_controls_df.shape[0]} controls\n{p_unknown_df.shape[0]} unknowns')

if __name__ == '__main__':
    main()
