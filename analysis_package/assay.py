
from pathlib import Path
from datetime import datetime
from . import maxlab_analysis as mla
import numpy as np
import pandas as pd
import h5py


class Assay:
    '''
    A container for all information about a single assay + some processing.
    path: Path
        The path to the folder that the assay (and all processing of the assay) is in. 
    assay_number: int
        the unique assay number (per project) assigned by the MaxLab system.
    assay_type: str
        the assay type. For now, I've only been working with NetworkAssays.
    chip_number: str
        the chip number
    date: datetime.date
        the date that the assay was conducted
    project: str
        the name of the project
    raw_data_path: Path
        the full path to the raw data
        
    '''
    def __init__(self, path, open_raw_h5 = True, build_raw_npy = True, build_spike_array = True, overwrite_raw_npy = False, overwrite_spike_array = False):
        '''
        Creates an Assay object from a path to a folder. Takes in a filepath to the folder with all data files in the form of a Path object or a string.
        the 'build' parameters tell you whether to create a file if it does not yet exist. They are either True or False.
        the 'overwrite' parameters, if true, ignore whether or not there already exists a spike array or raw npy file. build parameter must be true for this to take effect.
        '''
        self.path = Path(path)
        self.assay_number = None
        self.assay_type = None
        self.chip_number = None
        self.date = None
        self.project = None
        
        self.raw_data_path = Path(self.path, 'data.raw.h5', mmap_mode = 'r')
        if open_raw_h5:
            self.spike_df = mla.load_spikes_from_file(Path(self.raw_data_path), voltage_threshold=-10)
        
            with h5py.File(self.raw_data_path, "r") as h5_file:
                h5_object = h5_file['wells']['well{0:0>3}'.format(0)]['rec{0:0>4}'.format(0)]

                # Load settings from file
                self.lsb = h5_object['settings']['lsb'][0]
                self.sampling = h5_object['settings']['sampling'][0]
                self.spike_threshold = h5_object['settings']['spike_threshold'][0]
                self.gain = h5_object['settings']['gain'][0]
                self.hpf = h5_object['settings']['hpf'][0]
                self.mapping = pd.DataFrame(np.array(h5_object['settings']['mapping']))
                self.record_time = int(h5_file['assay']['inputs']['record_time'][0])

        self.analyses = dict()

        #make spike array
        build_spike_array_func = lambda full_filename: mla.bin_spike_data(self.spike_df, bin_size = 0.02, mode = 'binary', save = True, save_name = (full_filename))
        self.load_build_npy('spike_array', build_spike_array_func, build_spike_array, overwrite_spike_array)

        #make raw npy
        build_raw_npy_func = lambda full_filename: mla.recording_to_npy(self.raw_data_path, well_no = 0, recording_no = 0,  block_size = 40000, frames_per_sample = 16, save_name = Path(full_filename))
        self.load_build_npy('raw', build_raw_npy_func, build_raw_npy, overwrite_raw_npy)

        #get info on the assay
        try:
            self.assay_number = int(self.path.parts[-1]) #should this be an int or string?
            self.assay_type = self.path.parts[-2]
            self.chip_number = self.path.parts[-3]
            self.date = datetime.strptime(self.path.parts[-4], '%y%m%d').date()
            self.project = self.path.parts[-5]
        except IndexError:
            pass


    def load_build_npy(self, filename: str, build_func = None, build = True, overwrite = False):
        '''
        A function to both load and build a .npy file if the .npy file does not exist yet.
        the 'build' parameter tell you whether to create a file if it does not yet exist. Either True or False.
        the 'overwrite' parameter, if true, ignores whether or not there already exists a file of the name filename. If true, overwrites build parameter.
        build_func can only be none if build = False or the file already exists and overwrite = False.
        build_func should be written so that it takes in one parameter: a FULL filename where the file will be saved (with the preceeding path). 
        It should save the new numpy array as a .npy file as the filename specified. See examples in analysis_pipeline.py.
        '''
        if not overwrite:
            try:
                self.analyses[filename] = np.load(Path(self.path, filename + '.npy'), mmap_mode = 'r')
            except FileNotFoundError:
                if build:
                    print('building ' + filename + '.npy' + ' in ' + str(self.path))
                    build_func(Path(self.path, filename))
                    self.analyses[filename] = np.load(Path(self.path, filename + '.npy'), mmap_mode = 'r')
                else:
                    self.analyses[filename] = None
        else:
            print('building ' + filename + ' in ' + str(self.path))
            build_func(Path(self.path, filename))
            self.analyses[filename] = np.load(Path(self.path, filename + '.npy'), mmap_mode = 'r')

    def __str__(self):
        string = ''
        string += ('path: ' + str(self.path) + '\n')
        string += ('project: ' + str(self.project) + '\n')
        string += ('date: ' + str(self.date) + '\n')
        string += ('chip_number: ' + str(self.chip_number) + '\n')
        string += ('assay_type: ' + str(self.assay_type) + '\n')
        string += ('assay_number: ' + str(self.assay_number) + '\n')
        string += ('analyses: ' + str(self.analyses.keys()))

        return string
    
    def __repr__(self):
        return str(type(self)) + ': ' + str(self.path)

class NetworkAssay(Assay):
    pass
class ActivityScanAssay(Assay):
    pass
class StimulationAssay(Assay):
    pass



#First, find the scans in the filetree.

#The filestream goes 'project_name/date/chip_num/scan_type/assay_number
#The resulting folder has a file 'data.raw.h5' that has the raw data
#as well as an 'Analysis' folder.


if __name__ == "__main__":

    parent_folder = 'D:/'
    project_name = 'Summer_2023_Batch_2'

    all_network_scans = mla.load_assays_from_project(parent_folder, project_name)

    for assay in np.concatenate(list(all_network_scans.values())):
        assay.load_build_npy('scaled')
        assay.load_build_npy('pca')

    print(all_network_scans)

    print(all_network_scans['20439'][1])
    print(list(all_network_scans['20439'][1].path.glob('*.*')))


