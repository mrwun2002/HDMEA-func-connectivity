import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx
import sys
import os
import pandas as pd
import spycon
import glob

import time
import logging

#https://github.com/christiando/spycon
#requires pip install git+https://github.com/christiando/spycon
#https://github.com/pwollstadt/IDTxl
#requires pip install git+https://github.com/pwollstadt/IDTxl
from spycon.spycon_inference import SpikeConnectivityInference
import spycon_utils

from typing import Tuple

from sklearn.cluster import DBSCAN


from spycon.coninf import *

import traceback


sys.path.append("..")
from analysis_package import maxlab_analysis as mla



def load_data(path: str, well_no: int = 0, recording_no: int = 0, start_time: float = 0, end_time: float = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns a tuple of a spike dataframe and a mapping dataframe
    """
    
    spikes = mla.load_spikes_from_file(path, well_no, recording_no)
    mapping = mla.load_mapping(path, well_no, recording_no)

    # for i in spikes.index:
    #     if spikes.loc[i, "channel"] not in mapping["channel"]:
    #         spikes.drop(i)
    spikes = spikes.loc[spikes["channel"].isin(mapping["channel"]), :]

    spikes = spikes.loc[(spikes["time"] < end_time * 60) & (start_time * 60 < spikes["time"])].reset_index(drop = True)
    return spikes, mapping




def remove_synchronized_bursts(spike_df: pd.DataFrame, delta_t: float = 0.5, fraction_threshold: float = 0.1):
    burst_spike_df, _ = mla.find_synchronized_bursts(spike_df, plot_firing = False, delta_t = delta_t, fraction_threshold = fraction_threshold)
    return spike_df.drop(burst_spike_df.index, axis = 0).reset_index(drop = True)





def calculate_spike_freqs_and_amps(spike_df: pd.DataFrame, mapping_df: pd.DataFrame):
    grouped_counts = spike_df.groupby(["channel"])["channel"].count()
    grouped_mean_amps = spike_df.groupby(["channel"])["amplitude"].mean()

    mapping_df_with_stats = mapping_df.copy()
    for i in mapping_df_with_stats.index:
        try:
            mapping_df_with_stats.loc[i, "frequency"] =grouped_counts[mapping_df_with_stats.loc[i, "channel"]]/(max(spike_df["time"]) - min(spike_df["time"])) * 60 #In spikes per min
            mapping_df_with_stats.loc[i, "mean amp"] = np.abs(grouped_mean_amps.loc[mapping_df_with_stats.loc[i, "channel"]])
        except Exception as err:
            print("Error:" + str(err))
            mapping_df_with_stats.loc[i, "frequency"] = 0
            mapping_df_with_stats.loc[i, "mean amp"] = 0
    
    return mapping_df_with_stats





def trim_electrodes_by_percentile(spike_df: pd.DataFrame, mapping_df_with_stats: pd.DataFrame, frequency_percentile: float = 70, amplitude_percentile: float = 20):

    freq_colors = mapping_df_with_stats.loc[:, "frequency"]

    plt.subplot(321)
    plt.title("Spike frequencies, all electrodes")
    plt.scatter(mapping_df_with_stats.loc[:, "x"], mapping_df_with_stats.loc[:, "y"],  s = 10, c = mapping_df_with_stats.loc[:, "frequency"], vmin = min(freq_colors), vmax = max(freq_colors))

    plt.subplot(322)
    plt.title(f"Top {100 - frequency_percentile}% firing rate electrodes")
    freq_trimmed_indices = mapping_df_with_stats["frequency"] > np.percentile(mapping_df_with_stats["frequency"], frequency_percentile)
    plt.scatter(mapping_df_with_stats.loc[freq_trimmed_indices, "x"], mapping_df_with_stats.loc[freq_trimmed_indices, "y"],  s = 10, c = mapping_df_with_stats.loc[freq_trimmed_indices, "frequency"], vmin = min(freq_colors), vmax = max(freq_colors))
    plt.colorbar()


    plt.subplot(323)
    plt.title("Spike amplitudes, all electrodes")
    amp_colors = mapping_df_with_stats.loc[:, "mean amp"]
    plt.scatter(mapping_df_with_stats.loc[:, "x"], mapping_df_with_stats.loc[:, "y"],  s = 10, c = mapping_df_with_stats.loc[:, "mean amp"], norm=mpl.colors.LogNorm(vmin = min(amp_colors) + 1, vmax = max(amp_colors)))

    plt.subplot(324)
    plt.title(f"Top {100 - amplitude_percentile}% spike amplitude electrodes")
    amp_trimmed_indices = mapping_df_with_stats["mean amp"] > np.percentile(mapping_df_with_stats["mean amp"], amplitude_percentile)
    plt.scatter(mapping_df_with_stats.loc[amp_trimmed_indices, "x"], mapping_df_with_stats.loc[amp_trimmed_indices, "y"],  s = 10, c = mapping_df_with_stats.loc[amp_trimmed_indices, "mean amp"], norm=mpl.colors.LogNorm(vmin = min(amp_colors) + 1, vmax = max(amp_colors)))
    plt.colorbar()


    trimmed_indices = freq_trimmed_indices & amp_trimmed_indices
    plt.subplot(326)
    plt.title(f"Top {100 - amplitude_percentile}% spike amplitude electrodes, top {100 - frequency_percentile}% spike frequency")
    trimmed_mapping_df = mapping_df_with_stats.loc[trimmed_indices, :].reset_index(drop = True)
    plt.scatter(trimmed_mapping_df.loc[:, "x"], trimmed_mapping_df.loc[:, "y"],  s = 10)

    plt.tight_layout()

    trimmed_spike_df = spike_df.loc[spike_df["channel"].isin(trimmed_mapping_df.loc[:, "channel"]), :].reset_index(drop = True)

    return trimmed_spike_df, trimmed_mapping_df





def cluster_spikes_by_mapping(spike_df: pd.DataFrame, mapping_df: pd.DataFrame, eps = 50, min_samples = 1):
    clustering = DBSCAN(eps = 50, min_samples = 1).fit(mapping_df.loc[:, ["x", "y"]])
    labels = clustering.labels_
    grouped_mapping_df = mapping_df.copy()
    grouped_mapping_df["label"] = labels

    plt.title("Electrode groupings")
    plt.scatter(grouped_mapping_df.loc[:, "x"], grouped_mapping_df.loc[:, "y"],  s = 10,  c = labels)

    clustering_dict = dict()
    for i, chan in enumerate(grouped_mapping_df["channel"]):
        clustering_dict[chan] = labels[i]

    grouped_mapping_df = grouped_mapping_df.drop(["channel"], axis = 1).rename(columns = {"label": "channel"})
    grouped_mapping_df = grouped_mapping_df.groupby("channel").mean().reset_index(drop = True)

    grouped_spike_df = spike_df.copy()

    grouped_spike_df["original channel"] = grouped_spike_df["channel"]
    grouped_spike_df["channel"] = grouped_spike_df["channel"].replace(clustering_dict)

    grouped_spike_df.drop_duplicates(inplace = True, subset = ["time", "channel"])

    grouped_spike_df = grouped_spike_df.reset_index(drop = True)


    return grouped_spike_df, grouped_mapping_df



def infer_connectivity(spike_df: pd.DataFrame, con_method = Smoothed_CCG()):
    times = spike_df["time"].values
    ids = spike_df["channel"].values

    plt.scatter(times, ids, s=2, c=[[.4,.4,.4]])
    #plt.yticks(np.unique(ids)[::5])
    plt.xlim([0,60])
    #plt.xticks([0,15,30])
    plt.ylabel('IDs')
    plt.xlabel('Time [s]')
    plt.title('Raster data')


    #con_method = Smoothed_CCG() #This has an alpha parameter that can be changed
    spycon_result = con_method.infer_connectivity(times, ids)
    return spycon_result





def predict_network(data_path, save_path, inference_method: SpikeConnectivityInference = Smoothed_CCG(), well_no = 0, recording_no = 0, start_time = 0, end_time = 20, frequency_percentile = 70, amplitude_percentile = 20, remove_bursts = True):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler_1 = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler_1)
    handler_2 = logging.FileHandler(f"{save_path}log.log", mode = "w")
    logger.addHandler(handler_2)
    
    try:
        #Set up save location and logs and timing
        full_start = time.time()
        logging.info("Starting processing file " + data_path + ": " + time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(full_start)))
        logging.info(# TODO: include details of inference method params
            f"""\t-----------PARAMS-----------
\tdata_path: {data_path}
\twell_no: {well_no}
\trecording_no: {recording_no}
\tstart_time: {start_time}
\tend_time: {end_time}
\tinference_method: {type(inference_method)}
\t\tparams: {inference_method.params}
\tfrequency_percentile: {frequency_percentile}
\tamplitude_percentile: {amplitude_percentile}
\tremove_bursts: {remove_bursts}
\t----------------------------
            """
            )

        start = time.time()
        logging.info(f"\tLoading data: {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(start))}")
        spike_pd_df, mapping_pd_df = load_data(data_path, well_no = well_no, recording_no = recording_no, start_time = start_time, end_time = end_time)
        spike_pd_df.to_csv(f"{save_path}01_original_spike_df.csv", index = False)
        mapping_pd_df.to_csv(f"{save_path}02_original_mapping_df.csv", index = False)
        
        end = time.time()
        logging.info(f"\tComplete: {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(end))} (Duration = {end - start} s)")

        if remove_bursts:
            start = time.time()
            logging.info("\tRemoving bursts: " + time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(start)))
            spike_pd_df = remove_synchronized_bursts(spike_pd_df)
            logging.info(f"\tComplete: {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(end))} (Duration = {end - start} s)")
            spike_pd_df.to_csv(f"{save_path}03_burst_removed_spike_df.csv", index = False)
            
        
        
        start = time.time()
        logging.info("\tSelecting and clustering electrodes: " + time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(start)))
        mapping_pd_df = calculate_spike_freqs_and_amps(spike_pd_df, mapping_pd_df)
        mapping_pd_df.to_csv(f"{save_path}04_mapping_df_with_stats.csv", index = False)
        logging.info("\t\telectrode stats calculated")

        plt.figure(figsize = [10, 9])  
        trimmed_spike_df, trimmed_mapping_df = trim_electrodes_by_percentile(spike_pd_df, mapping_pd_df, frequency_percentile =  frequency_percentile, amplitude_percentile = amplitude_percentile)
        trimmed_spike_df.to_csv(f"{save_path}05_trimmed_spike_df.csv", index = False)
        trimmed_mapping_df.to_csv(f"{save_path}06_trimmed_mapping_df.csv", index = False)
        plt.savefig(f"{save_path}electrodes_percentiles.png")
        plt.close()
        logging.info("\t\telectrodes trimmed")
        
        plt.figure(figsize = (5, 4))
        grouped_spike_df, grouped_mapping_df = cluster_spikes_by_mapping(trimmed_spike_df, trimmed_mapping_df)
        grouped_spike_df.to_csv(f"{save_path}07_grouped_spike_df.csv", index = False)
        mapping_pd_df.to_csv(f"{save_path}08_grouped_mapping_df.csv", index = False)
        plt.savefig(f"{save_path}electrodes_clusters.png")
        plt.close()
        logging.info("\t\tspikes clustered")
        end = time.time()
        logging.info(f"\tComplete: {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(end))} (Duration = {end - start} s)")


        #TODO: Pass in learning parameters for each method
        start = time.time()
        logging.info("\tInferring connectivity: " + time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(start)))

        plt.figure(figsize = (6, 3))
        spycon_result = infer_connectivity(grouped_spike_df, con_method = inference_method)
        plt.savefig(f"{save_path}inference_raster.png")
        plt.close()

        spycon_result.save("spycon_network", save_path)
        end = time.time()
        logging.info(f"\tComplete: {time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(end))} (Duration = {end - start} s)")

        

        logging.info(f"\tProducing final graphs...")
        fig = plt.figure(figsize=(16,12))
        ax1 = fig.add_subplot(221)
        g1 = spycon_result.draw_graph(graph_type='stats', ax=ax1)
        pos1 = nx.circular_layout(g1)
        for i in grouped_mapping_df.index:
            pos1[i] = np.array(grouped_mapping_df.loc[i, ["x", "y"]] )

        ax1.set_title('Stats graph')
        ax2 = fig.add_subplot(222)
        g2 = spycon_result.draw_graph(graph_type='weighted', ax=ax2)
        pos2 = nx.circular_layout(g2)
        for i in grouped_mapping_df.index:
            pos2[i] = np.array(grouped_mapping_df.loc[i, ["x", "y"]])
        ax2.set_title('Inferred graph')

        ax3=fig.add_subplot(223)
        nx.draw(g1, pos1, with_labels=True)
        cmap = plt.get_cmap("inferno_r")
        weights = list(nx.get_edge_attributes(g1, "weight").values())
        min_weight, max_weight = np.amin(weights), np.amax(weights)
        norm = mpl.colors.Normalize(vmin=min_weight, vmax=max_weight)
        nx.draw(g1, pos1, with_labels=True,
                node_color="C1",
                edge_color=weights,
                edge_vmin=min_weight,
                edge_vmax=max_weight,
                edge_cmap=cmap,
                )
        plt.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax = plt.gca(), 
                    label="stats",
                )

        ax4=fig.add_subplot(224)
        cmap = plt.get_cmap("BrBG")
        weights = list(nx.get_edge_attributes(g2, "weight").values())
        max_weight = np.amax(np.absolute(weights))
        norm = mpl.colors.Normalize(vmin=min_weight, vmax=max_weight)
        nx.draw(g2, pos2, with_labels=True,
                node_color="C1",
                edge_color=weights,
                edge_vmin=-max_weight,
                edge_vmax=max_weight,
                edge_cmap=cmap,
                )
        plt.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax = plt.gca(), 
                    label="weights",
                )
        plt.savefig(f"{save_path}network_preview_1.png")
        plt.close()




        fig = plt.figure(figsize=(16,18))
        for num, thresh in enumerate([5, 10, 20, 40, 80]):
            spycon_result.set_threshold(thresh)
            ax = plt.subplot(3, 2, num+1)
            
            try:
                g = spycon_utils.draw_graph_with_mapping(spycon_result, grouped_mapping_df, "weighted", ax = ax)
            except Exception as err:
                print(err)
            plt.title("threshold = " + str(thresh) + ": " + str(len(g.edges)) + " edges")
        plt.savefig(f"{save_path}network_preview_2.png")
        plt.close()

        logging.info("Network prediction complete! Total duration: " + str(time.time() - full_start) +" s")

        logger.removeHandler(handler_1)
        logger.removeHandler(handler_2)
        del logger, handler_2, handler_1


    except Exception as err:
        plt.close('all')
        logging.exception("Exception occurred: %s\n", str(err))
        logger.removeHandler(handler_1)
        logger.removeHandler(handler_2)
        del logger, handler_2, handler_1
        return




def batch_predict_network(data_path: str, parent_folder: str, inference_method: SpikeConnectivityInference = Smoothed_CCG(), well_no = 0, recording_no = 0, start_time = 0, end_time = 20, frequency_percentile = 70, amplitude_percentile = 20, remove_bursts = True):
    assert type(parent_folder) is not list
    assert type(data_path) is not list

    args = [well_no, recording_no, start_time, end_time, frequency_percentile, amplitude_percentile, remove_bursts]
    arg_names = ["well_no", "recording_no", "start_time", "end_time", "frequency_percentile", "amplitude_percentile", "remove_bursts"]
    
    
    for i, param in enumerate(args):
        if type(param) is list:
            for param_element in param:
                new_parent_folder = f"{parent_folder}{arg_names[i]}={param_element}/"
                os.mkdir(new_parent_folder)
                new_args = args
                new_args[i] = param_element
                
                batch_predict_network(data_path, new_parent_folder, inference_method, *new_args)
            
            return
    
    predict_network(data_path, parent_folder, inference_method, *args)




            

    



if __name__ == '__main__':
    inference_methods = ["CoincidenceIndex", "Smoothed_CCG", "directed_STTC", "TE_IDTXL", "GLMPP", "GLMCC"]
    #default_alpha_values = [0.01, 0.01, 0.001, 0.01, 0.01, 0.01]
    #alpha_value_multipliers = [1]


    days = [30, 33, 34, 35, 36, 37, 38, 40, 41]
    days = [33, 34, 35, 36, 37, 38]
    #days = [30, 33]
    chip = "M07480"
    well_nos = range(6)
    pre_time = 20 #In theory, these can be extracted from events.
    train_time = 20
    post_time = 20

    remote_drive_path = "R:/"
    remote_drive_path = "/run/user/1000/gvfs/smb-share:server=rstoreint.it.tufts.edu,share=as_rsch_levinlab_wclawson01$/"
    parent_save_path = "HDMEA-func-connectivity/data/test_network_results/"
    parent_save_path = "/run/user/1000/gvfs/smb-share:server=rstoreint.it.tufts.edu,share=as_rsch_levinlab_wclawson01$/Experimental Data/nathan_senior_project_analysis/stim_removal_network_graphs_run_3/"
    
    for i, inference_method in enumerate(inference_methods):
        #default_alpha_value = default_alpha_values[i]
        #alpha_values = [m * default_alpha_value for m in alpha_value_multipliers]
        if inference_method == "CoincidenceIndex":
            con_method = CoincidenceIndex()
        elif inference_method == "Smoothed_CCG":
            con_method = Smoothed_CCG()
        elif inference_method == "directed_STTC":
            con_method = directed_STTC()
        elif inference_method == "TE_IDTXL":
            con_method = TE_IDTXL()
        elif inference_method == "GLMPP":
            con_method = GLMPP()
        elif inference_method == "GLMCC":
            con_method = GLMCC()
        else:
            con_method = None


        for day in days:
            for well_no in well_nos:
                file_path = f"{remote_drive_path}Experimental Data/Summer 2024/stim_removal/DIV{day}_stim_removal/{chip}/"
                print("")
                print(f"Starting day {day} well {well_no}")
                try:
                    file_path = glob.glob(file_path + "/*/")[-1] #date
                    file_path = glob.glob(file_path + "/*/")[-1] #trial
                    file_path = file_path + f"well{well_no}/"

                    file_name = f"DIV{day}_stim_removal_well_{well_no}.raw.h5"

                    recording_no = 0

                    save_path = f"{parent_save_path}{inference_method}/DIV{day}/well_no={well_no}/"
                    os.makedirs(save_path)

                except Exception as err:
                    print("\tError!", type(err).__name__, err)
                    print(traceback.format_exc())
                    continue

                batch_predict_network(file_path + file_name, save_path, inference_method= con_method, well_no = well_no, end_time = 20, remove_bursts = [True, False])

