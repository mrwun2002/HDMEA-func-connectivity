import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import re
import os
import sys
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from typing import List

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../HALnalysis")
import analysis_package as mla


class Burst:
    def __init__(self, time: float, raster_data: pd.DataFrame, mapping: pd.DataFrame, window_size: int, bin_duration: float, fraction_channels_firing: np.array, filtered_channels_firing: np.array):
        self.time = time
        self.raster_data = raster_data
        
        self.window_size = window_size #NOTE: THE FULL SIZE OF THE WINDOW IS 2 * window_size + 1 (one window before and one window after)
        

        self.mapping = mapping

        self.num_channels = len(raster_data.columns)
        self.bin_duration = bin_duration #in seconds
        self.fraction_channels_firing = fraction_channels_firing #this can be automatically calculated
        self.filtered_channels_firing = filtered_channels_firing #this can be automatically calculated?
        
        self.avg_x_positions = None
        self.xtplot = None #move this to be a part of plot?

    def plot(self, plot_avg_x_positions = True):
        fig = plt.figure()
        ax_dict = fig.subplot_mosaic('''
                                     aaaab
                                     '''
                                     , sharey = True)

        plt.sca(ax_dict["a"])
        plt.title(f"time = {self.time:.2f}")
        plt.xlabel("space (μm)")
        plt.ylabel("time (s)")

        if self.xtplot is None:
            print("building xt plot")
            self.build_xt_plot()

        plt.imshow(self.xtplot, extent = [0, 3850, self.time + (self.window_size) * self.bin_duration, self.time - (self.window_size) * self.bin_duration], aspect = 'auto')
        
        plt.plot(self.avg_x_positions, np.arange(self.time - self.window_size * self.bin_duration, self.time + (self.window_size + 0.5) * self.bin_duration, self.bin_duration), "g")


        plt.sca(ax_dict['b'])

        plt.plot(self.fraction_channels_firing, np.arange(self.time - self.window_size * self.bin_duration, self.time + (self.window_size + 0.5) * self.bin_duration, self.bin_duration))
        if not (self.filtered_channels_firing is None):
            plt.plot(self.filtered_channels_firing, np.arange(self.time - self.window_size * self.bin_duration, self.time + (self.window_size + 0.5) * self.bin_duration, self.bin_duration))

        plt.xlim([0, 1])
        plt.xlabel("Fraction of\nchannels firng")
        plt.tight_layout()
        plt.show() 

    def duration(self, thresh = 0.1) -> float:
        if not (self.filtered_channels_firing is None):
            firing = self.filtered_channels_firing
        else:
            firing = self.fraction_channels_firing

        start = None
        end = None
        for i, fraction_firing in enumerate(firing[0:self.window_size + 2]):
            if fraction_firing >= thresh and (start is None):
                start = i #Closest start time before the thresh
            elif fraction_firing < thresh and not (start is None):
                start = None
            
        for i, fraction_firing in enumerate(firing[self.window_size:]):
            if fraction_firing < thresh:
                end = i + self.window_size
                break
        
        if start is None or end is None:
            return None, None, None
        duration = (end - start) * self.bin_duration
        return duration, start, end

    def calculate_onset_offset(self, step = 1, thresh = 0.1) -> tuple:
        _, start, end = self.duration(thresh)
        offset = np.inf
        onset = -np.inf


        for i in range(start, self.window_size):
            
            change = self.fraction_channels_firing[i] - self.fraction_channels_firing[i-step]
            if onset < change:
                onset = change

        for i in range(self.window_size + step, end + step):
            change = self.fraction_channels_firing[i] - self.fraction_channels_firing[i-step]
            if offset > change:
                offset = change

        return (offset, onset)

    def max_height(self, thresh = 0.1):
        _, start, end = self.duration(thresh)
        return max(self.fraction_channels_firing[start: end])


    def direction(self, thresh = 0.1) -> float:
        _, start, _ = self.duration(thresh)

        if (start is None):
            return 0

        if self.avg_x_positions is None:
            self.build_xt_plot()
        return self.avg_x_positions[self.window_size] - self.avg_x_positions[start]
    
    def build_xt_plot(self, num_x_bins = 43) -> np.array:
        xt_plot = np.zeros((2 * self.window_size + 1, num_x_bins))
        avg_x_positions = np.empty(2*self.window_size + 1)
        for frame in range(-self.window_size, self.window_size + 1):
            x_positions = list()

            channels_with_spikes = list()
            for channel in self.raster_data.columns:
                
                if self.raster_data.loc[self.window_size + frame, channel]:
                    channels_with_spikes.append(channel)
            
            x_vals = self.mapping.loc[self.mapping["channel"].isin(channels_with_spikes), "x"]

            space_bin_counts, _ = np.histogram(x_vals, num_x_bins, (0, 3850))
            avg_x_positions[self.window_size + frame] = np.mean(x_vals) #this gets average x positions
            xt_plot[self.window_size + frame, :] = space_bin_counts


        self.xtplot = xt_plot
        self.avg_x_positions = avg_x_positions
        
        



def find_bursts(filepath, filename, well_no = 0, recording_no = 0, bin_duration = 0.01, num_x_bins = 43, window_size = 50, plot = True, burst_thresh = 0.3, stim_thresh = 0.8, chip_id = "", datapath = "data/wes_analysis/", filtered = True):
    mapping = mla.load_mapping(filepath + filename + ".raw.h5", well_no, recording_no)

    raster_data_filename = datapath + filename + f"_well_{well_no}_rec_{recording_no}_spike_raster_{bin_duration}s_{chip_id}.pkl" 
    spike_data_filename = datapath + filename + f"_well_{well_no}_rec_{recording_no}_spike_data_{bin_duration}s_{chip_id}.pkl"
    times_filename = datapath + filename + f"_well_{well_no}_rec_{recording_no}_times_{bin_duration}s_{chip_id}.npy"
    try: 
        raster_data = pd.read_pickle(raster_data_filename)
        spike_data = pd.read_pickle(spike_data_filename)
        times = np.load(times_filename)
    except FileNotFoundError:
        print ("files not found") #TODO:    Check if the filepath exists first.
        print("Binning data...")
        data = mla.load_spikes_from_file(filepath+ filename + ".raw.h5", well_no, recording_no)

        print(data)

        #data["frameno"] = data["frameno"] - data.loc[0, "frameno"]
        #FILTER OUT UNMAPPED CHANNELS
        #may not actually be necessary: Evan's bin spike data code does this for me.
        data = data.loc[data["channel"].isin(mapping["channel"]), :].reset_index(drop = True)

        raster_data, spike_data, times = mla.bin_spike_data(data, mapping, bin_size = bin_duration)
        raster_data.to_pickle(raster_data_filename)
        spike_data.to_pickle(spike_data_filename)
        np.save(times_filename, times)


    #Get total activity in each time bin, then smooth it out.
    fraction_channels_active = raster_data.to_numpy().mean(axis = 1)

    #Filter with gaussian convolution.
    filtered_fraction_channels_active = gaussian_filter1d(fraction_channels_active, 1)


    rms = np.sqrt(np.mean(filtered_fraction_channels_active**2))

    # peak_locs, properties = find_peaks(filtered_fraction_channels_active, rms * burst_thresh, distance = 30)
    if filtered:
        peak_locs, properties = find_peaks(filtered_fraction_channels_active, burst_thresh, distance = 30)
    else:
        peak_locs, properties = find_peaks(fraction_channels_active, burst_thresh, distance = 30)

    #REMOVE PEAKS THAT ARE TOO BIG (STIMULATIONS MOST LIKELY)
    peak_locs = peak_locs[properties["peak_heights"] < stim_thresh]



    if plot:
        plt.figure(figsize = (10, 10))
        plt.plot(times, fraction_channels_active)
        plt.plot(times, filtered_fraction_channels_active)
        #plt.hlines([rms * burst_thresh], 0, max(times), "r")
        plt.hlines([burst_thresh], 0, max(times), "r")
        plt.hlines([stim_thresh], 0, max(times), "r")
        plt.vlines(times[peak_locs], 0, 1, "g", alpha = 0.3)
        plt.show()



    burst_list = list()
    #We need: loc, windowsize, raster data,

    for i, loc in enumerate(peak_locs):
        if loc < window_size or loc > len(raster_data) - window_size - 1:
            continue
        print(f"burst found in bin {loc} (time {times[loc]:.2f} s)")
        burst_list.append(Burst(times[loc], raster_data.loc[loc-window_size:loc + window_size+1, :].reset_index(drop = True), mapping, window_size, bin_duration, fraction_channels_active[loc - window_size:loc + window_size + 1], filtered_fraction_channels_active[loc - window_size:loc + window_size + 1]))
    
    print("DONE")


    return burst_list