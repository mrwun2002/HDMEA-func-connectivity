import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import h5py
import numpy as np
import pandas as pd
import time
import math
import sys

from scipy.signal import find_peaks
from .assay import *

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

"""
This script shows how to open a raw data file and how to read and interpret the data. 
ATTENTION: The data file format is not considered stable and may change in the future.
"""

def __get_first_frame(filename, well_no, recording_no):
    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]
        groups = h5_object['groups']
        
        if len(list(groups)) > 0:
            first_frame_no = groups[next(iter(groups))]['frame_nos'][0]
        else: #take frame from spike data
            spike_dataset = h5_object['spikes']
            
            spike_array = np.array(spike_dataset)
            spike_pd_dataset = pd.DataFrame(spike_array)

            first_frame_no = spike_pd_dataset['frameno'][0]
    
    return first_frame_no

def load_from_file_by_frames(filename: str,  start_frame: int, block_size: int, well_no: int = 0, recording_no:int = 0,  frames_per_sample:int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets a numpy array of the voltage traces across all channels by time, from a raw h5 file.
    The native sample rate for the M1 is 20000 samples per second.
    The frames between ``start_frame`` and ``start_frame + block_size * frames_per_sample`` will be selected, every ``frames_per_sample`` frames.

    :param filename: The name of the h5 file
    :type filename: ``str``
    :param start_frame: The frame to start at for loading data.
    :type start_frame: ``int``
    :param block_size: The number of samples to load. Maximum allowed block size is 40000. 
    :type block_size: ``int``
    :param well_no: The well number, defaults to 0
    :type well_no: ``int``, optional
    :param recording_no: The recording number, defaults to 0
    :type recording_no: ``int``, optional
    :param frames_per_sample: The sampling rate, in frames per sample, defaults to 1
    :type frames_per_sample: ``int``, optional
    :return: Tuple of the voltage traces, an array of time values, and the channel numbers.
    :rtype: ``Tuple[np.ndarray, np.ndarray, np.ndarray]``
    """


    # The maximum allowed block size can be increased if needed,
    # However, if the block size is too large, handling of the data in Python gets too slow.

    #sample rate is 20000 samples per second
    max_allowed_block_size = 40000
    assert(block_size<=max_allowed_block_size)

    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]

        # Load settings from file
        lsb = h5_object['settings']['lsb'][0]
        #lsb = 1
        sampling = h5_object['settings']['sampling'][0]

        # compute time vector
        #stop_frame = start_frame+block_size
        time = np.arange(start_frame,start_frame+block_size * frames_per_sample,frames_per_sample) / sampling

        # Load raw data from file
        groups = h5_object['groups']
        #print(groups)
        group0 = groups[next(iter(groups))]

        channels = np.array(group0["channels"])

        try:
            return group0['raw'][:,start_frame:start_frame+block_size*frames_per_sample:frames_per_sample].T * lsb , time, channels
        except OSError as err:
            print(err)
            print("OSError thrown. you gotta run this on the maxlab computer :(")
            sys.exit()



def recording_to_csv(filename: str, well_no: int = 0, recording_no: int = 0,  block_size: int = 40000, frames_per_sample: int = 16, csv_name: int | None = None, delimiter: str = ','):
    """
    Save raw data to a readable .csv file, in blocks.

    :param filename: The name of the file to read.
    :type filename: ``str``
    :param well_no: The well number, defaults to 0
    :type well_no: ``int``, optional
    :param recording_no: The recording number, defaults to 0
    :type recording_no: ``int``, optional
    :param block_size: Number of frames to grab in each block, defaults to 40000
    :type block_size: ``int``, optional
    :param frames_per_sample: Sample rate, defaults to 16
    :type frames_per_sample: ``int``, optional
    :param csv_name: The name with which to save the array as a .csv file. If None, saves the file to the name ``data/`` + the name of the file. defaults to None
    :type csv_name:  ``int | None ``, optional
    :param delimiter: The delimiter used in the csv file, defaults to ','
    :type delimiter: str, optional
    """
    #get channel numbers, number of frames
    #test
    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]
        groups = h5_object['groups']
        group0 = groups[next(iter(groups))]
        
        (num_channels, num_frames) = np.shape(group0['raw'])

    column_headers = ['time'] + list(np.arange(1, num_channels + 1).astype(str))

    if csv_name == None:
        csv_name = 'data/' + filename + '.csv'
    
    #df = pd.DataFrame(columns = column_headers)
    #df.to_csv(csv_name, mode = 'w', index = False, header = True)

    with open(csv_name, 'w') as csvfile:
        np.savetxt(csvfile, [], header = ' '.join(column_headers), delimiter=delimiter)

    
    for block_start in np.arange(0, num_frames, block_size * frames_per_sample): 
        #block_end = min(block_start + block_size * frames_per_sample, num_frames)
        frames_to_end = num_frames - block_start
        print('writing frames ' + str(block_start) + ' to '  + str(block_start + min(block_size * frames_per_sample, frames_to_end)) + ' out of ' + str(num_frames))
        X, t = load_from_file_by_frames(filename, block_start, min(block_size, frames_to_end//frames_per_sample), well_no = well_no, recording_no = recording_no, frames_per_sample = frames_per_sample)
        full_arr = np.hstack((np.reshape(t, (-1, 1)), X))
        with open(csv_name, 'a') as csvfile:
            np.savetxt(csvfile, full_arr, delimiter = delimiter, fmt = '%.6g')


def recording_to_npy(filename: str, well_no: int = 0, recording_no: int = 0,  block_size: int = 40000, frames_per_sample: int = 16, save_name: int | None = None) -> None:
    """
    Save raw data to a readable .npy file, in blocks.

    :param filename: The name of the file to read.
    :type filename: ``str``
    :param well_no: The well number, defaults to 0
    :type well_no: ``int``, optional
    :param recording_no: The recording number, defaults to 0
    :type recording_no: ``int``, optional
    :param block_size: Number of frames to grab in each block, defaults to 40000
    :type block_size: ``int``, optional
    :param frames_per_sample: Sample rate, defaults to 16
    :type frames_per_sample: ``int``, optional
    :param save_name: The name with which to save the npy array as a .npy file. If None, saves the file to the name ``data/`` + the name of the file. defaults to None
    :type save_name:  ``int | None ``, optional
    """
    #get channel numbers, number of frames
    #16 frames per sample at 20000 frames per second = 1250 samples per second
    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]
        groups = h5_object['groups']
        group0 = groups[next(iter(groups))]
        
        (num_channels, num_frames) = np.shape(group0['raw'])

    if save_name == None:
        save_name = 'data/' + filename
    
    arr = np.zeros((int(num_frames/frames_per_sample), num_channels + 1), 'float32')
    
    for i, block_start in enumerate(np.arange(0, num_frames, block_size * frames_per_sample)): 
        #block_end = min(block_start + block_size * frames_per_sample, num_frames)
        frames_to_end = num_frames - block_start
        print('writing frames ' + str(block_start) + ' to '  + str(block_start + min(block_size * frames_per_sample, frames_to_end)) + ' out of ' + str(num_frames))
        X, t = load_from_file_by_frames(filename, block_start, min(block_size, frames_to_end//frames_per_sample), well_no = well_no, recording_no = recording_no, frames_per_sample = frames_per_sample)
        full_arr = np.hstack((np.reshape(t, (-1, 1)), X))
        del X, t
        arr[i * block_size:i * block_size + min(block_size, frames_to_end//frames_per_sample), :] = full_arr
        
        # with open(csv_name, 'a') as csvfile:
        #     np.savetxt(csvfile, full_arr, delimiter = delimiter, fmt = '%.6g')

    np.save(save_name, arr)

def load_spikes_from_file(filename: str, well_no: int = 0, recording_no: int = 0, voltage_threshold: float | None = None) -> pd.DataFrame:
    """
    Returns a pd dataset of the spike data.

    :param filename: The filename of the h5 file holding the data.
    :type filename: ``str``
    :param well_no: The well number, defaults to 0
    :type well_no: ``int``, optional
    :param recording_no: The recording number, defaults to 0
    :type recording_no: ``int``, optional
    :param voltage_threshold: Should be a negative number. If the spike amplitude is less than the voltage threshold (of greater magnitude), the spike will be counted. If None, the built-in spike detection threshold is used. defaults to None.
    :type voltage_threshold: ``float | None``, optional
    :return: The pandas dataframe of the spike data.
    :rtype: ``pd.DataFrame``
    """
    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]

        # Load settings from file
        sd_threshold = h5_object['settings']['spike_threshold'][0]
        native_sampling_rate = h5_object['settings']['sampling'][0]

        

        spike_dataset = h5_object['spikes']
        
        spike_array = np.array(spike_dataset)
        spike_pd_dataset = pd.DataFrame(spike_array)
        
        groups = h5_object['groups']
        first_frame_no = __get_first_frame(filename, well_no, recording_no) #NOTE: This can sometimes produce negative values for spike times.
        #first_frame_no = spike_pd_dataset['frameno'][0]

        spike_pd_dataset['frameno'] = spike_pd_dataset['frameno'] - first_frame_no

        if voltage_threshold != None:
            spike_pd_dataset = spike_pd_dataset.loc[spike_pd_dataset['amplitude'].le(voltage_threshold)]

        spike_pd_dataset['frameno'] = spike_pd_dataset['frameno'].multiply(1/native_sampling_rate)
        
        spike_pd_dataset.rename(columns = {'frameno':'time'}, inplace = True)

        
        return spike_pd_dataset


#question: should I delete frameno and frameno_adjuted?
def load_events(filename: str, well_no: int = 0, recording_no: int = 0) -> pd.DataFrame:
    """
    Returns a pd dataset of the events in the recording.

    :param filename: The filename of the h5 file holding the data.
    :type filename: ``str``
    :param well_no: The well number, defaults to 0
    :type well_no: ``int``, optional
    :param recording_no: The recording number, defaults to 0
    :type recording_no: ``int``, optional
    :return: A pandas dataframe of the events, with timestamps.
    :rtype: ``pd.DataFrame``
    """
    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]
        first_frame =__get_first_frame(filename, well_no, recording_no)
        native_sampling_rate = h5_object['settings']['sampling'][0]

        events = h5_object['events']
        events_arr= np.array(events)
        events_dataset = pd.DataFrame(events_arr)
    
    events_dataset["frameno_adjusted"] = events_dataset["frameno"] - first_frame
    events_dataset["time"] = events_dataset["frameno_adjusted"]/native_sampling_rate
    events_dataset["eventmessage_decoded"] = events_dataset["eventmessage"].apply(lambda x: x.decode("utf8"))

    return events_dataset

def load_mapping(filename: str, well_no: int = 0, recording_no: int = 0):
    """
    Returns a pd dataset of the mapping between channels, electrodes, and spatial positions.

    :param filename: The filename of the h5 file holding the data.
    :type filename: ``str``
    :param well_no: The well number, defaults to 0
    :type well_no: ``int``, optional
    :param recording_no: The recording number, defaults to 0
    :type recording_no: ``int``, optional
    :return: A pandas dataframe of the mappings.
    :rtype: ``pd.DataFrame``
    """
    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]
        mapping = pd.DataFrame(np.array(h5_object["settings"]["mapping"]))

    return mapping


    
def bin_spike_data(spike_df: pd.DataFrame, mapping: pd.DataFrame, bin_size: float =0.01, mode: str ='binary') -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Takes in a spike data dataframe and a mapping dataframe, bins the spike data by time according to the channels specified in the mapping.
    
    :param spike_df: The dataframe of spikes.
    :type spike_df: pd.DataFrame
    :param mapping: The pandas dataframe corresponding to the mapping from channels to electrodes.
    :type mapping: pd.DataFrame
    :param bin_size: The size of time bins, in seconds. defaults to 0.01
    :type bin_size: float, optional
    :param mode: mode must be 'binary' or 'count'. Whether multiple spikes in the same bin are summed together, or whether data is simply binary. defaults to 'binary'
    :type mode: str, optional
    :return: A tuple of the binned data (pandas dataframe), the spike data with an additional column corresponding to the bin number, and a 1d np array of the same length as the binned data storing the times corresponding to each bin.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]
    """
    # check to see if any channels have multiple rows in the mapping
    channel_counts = mapping['channel'].value_counts()
    duplicate_channels = channel_counts[channel_counts > 1]
    if not duplicate_channels.empty:
        print(f"Error: Channels with multiple rows in mapping - this is very bad!: {duplicate_channels.index.tolist()}")
        #raise ValueError(f"Error: Channels with multiple rows in mapping - this is very bad!: {duplicate_channels.index.tolist()}")

    assert mode in ['binary', 'count'], "mode must be binary or count."
    last_spike_time = spike_df['time'].max()
    first_spike_time = spike_df['time'].min()
    total_duration = last_spike_time - first_spike_time
    #num_bins = int(np.round(total_duration / bin_size)) + 1
    mapped_channels = mapping['channel'].unique()
    num_bins = int(np.floor(1+last_spike_time/bin_size) - np.floor(first_spike_time/bin_size))
    binned_data_array = np.zeros((num_bins, len(mapped_channels)), dtype=int)
    spike_data = spike_df.copy()
    bin_ids = np.full(spike_data.shape[0], -1)

    for index, row in spike_df.iterrows():
        bin_index = int(np.floor(row['time'] / bin_size) - np.floor(first_spike_time / bin_size))
        if row['channel'] in mapped_channels:
            channel_index = np.where(mapped_channels == row['channel'])[0][0]
            if mode == 'binary':
                binned_data_array[bin_index, channel_index] = 1
            elif mode == 'count':
                binned_data_array[bin_index, channel_index] += 1
        bin_ids[index] = bin_index

    spike_data['bin_id'] = bin_ids

    binned_data_df = pd.DataFrame(binned_data_array, columns=mapped_channels)

    times = np.arange(np.floor(first_spike_time / bin_size) * bin_size, 
                      np.floor(first_spike_time / bin_size) * bin_size + num_bins * bin_size - bin_size/2, 
                      bin_size)

    spike_df.reset_index(drop=True, inplace=True)

    assert len(times) == num_bins, print(f"Times: {len(times)}, Num bins: {num_bins}")

    return binned_data_df, spike_data, times

# def spike_array_from_file(filename, save = True, save_name = None, **kwargs):
#     """
#     Runs load_spikes_from_file() and then bin_spike_data() on the result. See those for documentation on parameters.
#     Returns a sparse numpy array with one axis as time and the other axis as channels with data on the spikes that occur within each time bin.
#     """
#     spike_df = load_spikes_from_file(filename, **kwargs)
#     arr, spike_data = bin_spike_data(spike_df, **kwargs)
#     if save:
#         if save_name == None:
#             save_name = filename + '.binned_spikes'

#       np.save(save_name, arr)

#     return arr, spike_data


def find_synchronized_bursts(df: pd.DataFrame, delta_t = 0.05, fraction_threshold = None, threshold_std_multiplier = 4, plot_firing = False): #TODO: make fraction threshold dependent upon distribution of numbers of neurons firing in a certain time delta
    '''
    Takes in a pd dataframe of spike data, a percentage threshold for bursts to be considered "synchronized", and a time delta (measured in seconds) in which to search for synchronized bursts.
    Returns a pd dataframe containing just the spikes in the synchronized bursts as well as a dataframe containin the start times of each burst, to the lowest time delta divided by two, and the number of channels active during each burst.
    '''
    num_channels = df['channel'].nunique() #Do we want this to be the number of active channels? the largest channel number?

    max_time = max(df['time'])
    min_time = min(df['time'])

    synchronized_bursts_df = pd.DataFrame(columns=df.columns)

    #look within a range of times, if a spike exists within that range, add the channel to a set of channels. Count the number of elements in that set.
    
    start_times = np.arange(min_time, max_time, delta_t/2)
    fraction_firing_channels = np.zeros_like(start_times, dtype=float)        


    for i, start_time in enumerate(start_times):

        entries_in_range = df.loc[df['time'].ge(start_time) & df['time'].lt(start_time + delta_t)]

        fraction_firing_channels[i] = (entries_in_range['channel'].nunique())/num_channels

    if fraction_threshold == None:
        fraction_threshold = threshold_std_multiplier*np.std(fraction_firing_channels) + np.mean(fraction_firing_channels)
        #print(fraction_threshold)
    
    (burst_indeces, burst_properties) = find_peaks(fraction_firing_channels, height = fraction_threshold)
    #print(burst_indeces)
    burst_times = start_times[burst_indeces]

    burst_times_df = pd.DataFrame(list(zip(burst_times, burst_properties['peak_heights'])), columns = ['time', 'fraction channels active'])

    if plot_firing:
        plt.figure()
        plt.plot(start_times, fraction_firing_channels)
        plt.xlabel('Time (s)')
        plt.ylabel('Number of channels')
        plt.hlines(fraction_threshold, 0, max_time, 'red')
        plt.show()

    for burst_time in burst_times:
        entries_in_range = df.loc[df['time'].ge(burst_time) & df['time'].lt(burst_time + delta_t)]
        synchronized_bursts_df = pd.concat([synchronized_bursts_df, entries_in_range])

    return synchronized_bursts_df, burst_times_df





def animate_pca(filestem, start_time, end_time, animation_framerate = 10, recording_framerate = 1250, speed_multiplier = 1, points_per_animation_frame = None, data_source = None, save_gif = True, save_name = None, reduce_memory_usage = False):
    '''
    Animates the first 3 axes of pca.
    filestem is the part of the data source before '.data.raw.h5'.
    start_time and end_time are in seconds. 
    animation_framerate and recording_framerate are in Hz.
    points_per_animation_frame defaults to, and maxes out at, recording_framerate * speed_multiplier / animation_framerate. 
    The passed-in value must be equal to this maximum points_per_animation frame value divided by an integer.
    data_source defaults to filestem + '.data.raw.h5'.
    save_name defaults to filestem + "_animation_" + str(start_time) + "-" + str(end_time) + "s_" + str(speed_multiplier) + "x_speed_" + str(points_per_animation_frame) + "_pts_per_" + str(animation_framerate) + "s"
    '''

    recording_frames_per_animation_frame = recording_framerate * speed_multiplier / animation_framerate

    if points_per_animation_frame == None:
        points_per_animation_frame = recording_frames_per_animation_frame

    recording_frames_per_animation_frame_subsample_rate = int(recording_frames_per_animation_frame/points_per_animation_frame)
    

    assert math.isclose(recording_frames_per_animation_frame_subsample_rate, recording_frames_per_animation_frame/points_per_animation_frame), "(recording_framerate * speed_multiplier / animation_framerate) / points_per_animation_frame must be an integer. \n (recording_framerate * speed_multiplier / animation_framerate) = " + str(recording_frames_per_animation_frame)

    if data_source == None: 
        data_source = filestem + ".data.raw.h5"
    if save_name == None:
        save_name = 'animations/' + filestem + "_animation_" + str(start_time) + "-" + str(end_time) + "s_" + str(speed_multiplier) + "x_speed_" + str(points_per_animation_frame) + "_pts_per_" + str(animation_framerate) + "s"

    data_from_npy = np.load(data_source + '.npy', mmap_mode = 'r', )

    #scale data
    t = data_from_npy[:, 0]

    if reduce_memory_usage:
        X = data_from_npy[:, 1::5]
    else:
        X = data_from_npy[:, 1::]
    Y = load_spikes_from_file(data_source, 0, 0, -10)
    Y_synchronized, spike_times = find_synchronized_bursts(Y)


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = 6

    pca = PCA(n_components)
    X_pca = pca.fit_transform(X_scaled)

    pc1_lims = [np.min(X_pca[:, 0]), np.max(X_pca[:, 0])]
    pc2_lims = [np.min(X_pca[:, 1]), np.max(X_pca[:, 1])]
    pc3_lims = [np.min(X_pca[:, 2]), np.max(X_pca[:, 2])]

    start_time_as_frame = start_time * recording_framerate
    end_time_as_frame = end_time * recording_framerate


    def animate_func(num):
        for ax in ax_dict.values():
            ax.clear()

        ax_dict['a'].set_xlim(pc1_lims)
        ax_dict['a'].set_ylim(pc2_lims)
        ax_dict['b'].set_xlim(pc1_lims)
        ax_dict['b'].set_ylim(pc3_lims)
        ax_dict['c'].set_xlim(pc1_lims)
        ax_dict['c'].set_ylim(pc3_lims)

        ax_dict['a'].set_xlabel('Principal component 1')
        ax_dict['a'].set_ylabel('Principal component 2')

        ax_dict['b'].set_xlabel('Principal component 1')
        ax_dict['b'].set_ylabel('Principal component 3')

        ax_dict['c'].set_xlabel('Principal component 2')
        ax_dict['c'].set_ylabel('Principal component 3')

        s1 = ax_dict['a'].scatter(X_pca[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate, 0], X_pca[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate, 1], c = t[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate], s = 2, alpha = 0.5, vmin = t[start_time_as_frame], vmax = t[end_time_as_frame])


        s2 = ax_dict['b'].scatter(X_pca[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate, 0], X_pca[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate, 2], c = t[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate], s = 2, alpha = 0.5, vmin = t[start_time_as_frame], vmax = t[end_time_as_frame])

        s3 = ax_dict['c'].scatter(X_pca[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate, 1], X_pca[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate, 2], c = t[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate], s = 2, alpha = 0.5, vmin = t[start_time_as_frame], vmax = t[end_time_as_frame])



        ax_dict['z'].scatter(Y[(Y['time'] < end_time) & (Y['time'] >= start_time)]['time'], Y[(Y['time'] < end_time) & (Y['time'] >= start_time)]['channel'], 0.5, c = Y[(Y['time'] < end_time) & (Y['time'] >= start_time)]['time'])
        #plt.scatter(Y_synchronized['frameno'], Y_synchronized['channel'], 1, 'r')
        ax_dict['z'].set_xlabel('Time (s)')
        ax_dict['z'].set_ylabel('Channels')
        ax_dict['z'].vlines(spike_times[(spike_times['time'] < end_time) & (spike_times['time'] >= start_time)]['time'], 0, max(Y['channel']), 'red', alpha=0.5)
        current_time = num/recording_framerate
        #s4 = ax_dict['z'].scatter(Y[Y['time'] < current_time]['time'], Y[Y['time'] < current_time]['channel'], 0.5, c = Y[Y['time'] < current_time]['time'])
        l = ax_dict['z'].vlines(current_time, 0, max(Y['channel']), 'green')

        plt.tight_layout()

        return s1, s2, s3, l#, s4

    fig = plt.figure(figsize = (12, 8))

    ax_dict = fig.subplot_mosaic(
        """
        abc
        zzz
        """
    )


    all_animation_frames = np.arange(start_time_as_frame, end_time_as_frame, recording_frames_per_animation_frame, dtype = int)

    title = fig.suptitle('Principal axes')

    animation = FuncAnimation(fig, animate_func, interval = 1000/animation_framerate, frames = all_animation_frames, blit = True, repeat = False)

    #plt.show()

    if save_gif:
        save_start_time = time.time()
        animation.save(save_name + '.gif', writer = PillowWriter(fps=60))
        print('gif saved')
        print('time taken: ' + str(time.time() - save_start_time))
    
    return fig

def load_assays_from_project(parent_folder, project_name, chips = set(), open_raw_h5 = True, build_raw_npy = True, build_spike_array = True, overwrite_raw_npy = False, overwrite_spike_array = False):
    '''
    For use with the built-in file structure of the MaxLab system. Takes in a parent folder (the location where all the projects are),
    the project name, and the chips that are to be looked at. Returns a dictionary of Assay objects indexed by the chip number.
    By default, chips is an empty set, which looks at all chips in the project.
    build_raw_npy and build_spike_array default to True, indicating that the raw npy array and spike array files will be saved into the folder if they do not exist yet.
    '''
    search_folder = Path(parent_folder, project_name)

    if not chips:
        #get just the chip number. I feel like theres probs a better way to do this but idk
        chips = set(chip_folder_path.parts[-1] for chip_folder_path in search_folder.glob("*/*"))
        #all_network_scans = list(search_folder.glob("*/*/Network/*/data.raw.h5"))

    all_assays = dict()
    for chip in chips:
        all_assays[chip] = list(NetworkAssay(raw_data.parent, open_raw_h5, build_raw_npy, build_spike_array, overwrite_raw_npy, overwrite_spike_array) for raw_data in search_folder.glob("*/" + str(chip) + "/Network/*/data.raw.h5"))
    
    return all_assays


if __name__ == "__main__":
    pass
