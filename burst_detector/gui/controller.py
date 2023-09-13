from main_window import MainWindow, _create_dock
from recording import Recording
from views import ClusterView, SimilarityView, SimLoadView, WaveformView, CorrelogramView, FiringRateView, ISIView, ProbeView
from PyQt5.QtWidgets import QFileDialog, QApplication, QAbstractItemView
from PyQt5.QtCore import Qt, QItemSelectionModel
from pyqtgraph import TableWidget, GraphicsLayoutWidget
from pyqtgraph.dockarea import DockLabel
import sys
import os
import burst_detector as bd
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import ToTensor
import threading


class Controller(object):
    """
    Controls GUI, links GUI and Recording classes, does core functions
    """
    def __init__(self):
        # load recording
        self.gui = self.create_gui()
        self.load_recording(True)

        os.makedirs(os.path.join(self.recording.ks_dir, "automerge"), exist_ok=True)

        # create views
        self.create_cluster_view()
        self.check_create_sim()
        
        self.create_wf_view()
        self.create_ccg_view()
        self.create_fr_view()
        self.create_isi_view()
        self.create_prb_view()

        self.sim_done = False


    def create_gui(self):
        gui = MainWindow(controller=self, name="SpECtr-GUI")
        return gui


    #TODO: cache curent/recent recordings
    def load_recording(self, new_sess=False):
        success = False
        while not success:
            try:
                filename,_ = QFileDialog.getOpenFileName()
                if not filename:  # if user presses cancel, stop trying to load
                    return
                recording = Recording(filename)
                success = True
            except IndexError as e:         # params file is not in the expected format
                # print("Given params file is not in the correct format!")
                print(e)
            except KeyError as e:           # params file is missing a required parameter
                print(e)
            except FileNotFoundError as e:  # a required file doesn't exit
                print(e)
        self.recording = recording
        self.params = self.recording.params
        if not new_sess:
            self.update_metrics(self.recording.cluster_metrics)

    #TODO: let autoencoder save spike index

    def check_create_sim(self):
        #check to see if spike_latents/ae_sim exists before training a new autoencoder
        if os.path.isfile(os.path.join(self.recording.ks_dir, "automerge", "ae_sim.npy")) and \
            os.path.isfile(os.path.join(self.recording.ks_dir, "automerge", "spk_lat_peak.npy")) and \
            os.path.isfile(os.path.join(self.recording.ks_dir, "automerge", "lat_mean.npy")) and \
            os.path.isfile(os.path.join(self.recording.ks_dir, "automerge", "spk_lab.npy")):

            print("Loading spike latents from file")
            self.ae_sim = np.load(os.path.join(self.recording.ks_dir, "automerge", "ae_sim.npy"))
            self.spk_lat_peak = np.load(os.path.join(self.recording.ks_dir, "automerge", "spk_lat_peak.npy"))
            self.lat_mean = np.load(os.path.join(self.recording.ks_dir, "automerge", "lat_mean.npy"))
            self.spk_lab = np.load(os.path.join(self.recording.ks_dir, "automerge", "spk_lab.npy"))

            print("Loading autoencoder from file")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.net = bd.autoencoder.CN_AE().to(device)      

            if (self.params['model_path'] is not None) and (os.path.isfile(self.params['spikes_path'])):
                self.net.load_state_dict(torch.load(self.params['model_path']))        
            elif os.path.isfile(os.path.join(self.recording.ks_dir, "automerge", "autoencoder.pt")):                       
                self.net.load_state_dict(torch.load(os.path.join(self.recording.ks_dir, "automerge", "autoencoder.pt")))
            self.net.eval()

            self.create_sim_view()

            return 
        
        self.create_sim_load_view()
        thread = threading.Thread(target=self.train_autoencoder)
        thread.start()


    def train_autoencoder(self):
         # extract spikes for training autoencoder if they don't already exist
        if (self.params['spikes_path'] is not None) and (os.path.isdir(self.params['spikes_path'])):
            spk_fld = self.params['spikes_path']
            print("Loaded spikes from %s" % spk_fld)
            self.sim_view.extractProg.setValue(100)
            self.sim_view.extractLbl.setText("Loaded spikes from file.")
        elif os.path.isdir(os.path.join(self.recording.ks_dir, "automerge", "spikes")):
            spk_fld = os.path.join(self.recording.ks_dir, "automerge", "spikes")
            print("Loaded spikes from %s" % spk_fld)
            self.sim_view.extractProg.setValue(100)
            self.sim_view.extractLbl.setText("Loaded spikes from file.")
        else:
            spk_fld = os.path.join(self.recording.ks_dir, "automerge", "spikes")
            ci = {
                'times': self.recording.sp_times,
                'times_multi': self.recording.cl_times,
                'clusters': self.recording.clusters,
                'counts': self.recording.counts,
                'labels': self.recording.cl_labels,
                'mean_wf': self.recording.mean_wf
            }
            gti = {
                'spk_fld': spk_fld,
                'pre_samples': self.params['ae_pre'],
                'post_samples': self.params['ae_post'],
                'num_chan': self.params['ae_chan'],
                'noise': self.params['ae_noise'],
                'for_shft': self.params['ae_shft']
            }
            print("No autoencoder snippet folder detected, extracting snippets to %s" % spk_fld)
            bd.autoencoder.generate_train_data(
                self.recording.data, 
                ci, 
                self.recording.channel_pos, 
                gti, 
                self.params,
                label=self.sim_view.extractLbl,
                prog=self.sim_view.extractProg
            )
            print("Finished extracting snippets.")

        # train autoencoder if it doesn't already exist
        if (self.params['model_path'] is not None) and (os.path.isfile(self.params['spikes_path'])):
            print("Loaded autoencoder from %s" % os.path.join(self.recording.ks_dir, "automerge", "autoencoder.pt"))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = bd.autoencoder.CN_AE().to(device)                         
            net.load_state_dict(torch.load(self.params['model_path']))
            net.eval()
            spk_data = bd.autoencoder.SpikeDataset(os.path.join(spk_fld, 'labels.csv'), spk_fld, ToTensor())
            self.sim_view.trainProg.setValue(100)
            self.sim_view.trainLbl.setText("Loaded autoencoder from file.")
        elif os.path.isfile(os.path.join(self.recording.ks_dir, "automerge", "autoencoder.pt")):
            print("Loaded autoencoder from %s" % os.path.join(self.recording.ks_dir, "automerge", "autoencoder.pt"))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = bd.autoencoder.CN_AE().to(device)                         
            net.load_state_dict(torch.load(os.path.join(self.recording.ks_dir, "automerge", "autoencoder.pt")))
            net.eval()
            spk_data = bd.autoencoder.SpikeDataset(os.path.join(spk_fld, 'labels.csv'), spk_fld, ToTensor())
            self.sim_view.trainProg.setValue(100)
            self.sim_view.trainLbl.setText("Loaded autoencoder from file.")
        else:
            print("Training autoencoder...")
            net, spk_data = bd.train_ae(
                spk_fld, 
                self.recording.counts, 
                do_shft=self.params['ae_shft'],
                num_epochs=self.params['ae_epochs'], 
                pre_samples=self.params['ae_pre'], 
                post_samples=self.params['ae_post'],
                label=self.sim_view.trainLbl,
                prog=self.sim_view.trainProg
            )
            torch.save(net.state_dict(), os.path.join(self.recording.ks_dir, "automerge", "autoencoder.pt"))
            print("Autoencoder saved to %s" % os.path.join(self.recording.ks_dir, "automerge", "autoencoder.pt"))
            net.eval()

        # calculate spike latents and ae_similarity
        self.ae_sim, self.spk_lat_peak, self.lat_mean, self.spk_lab = bd.stages.calc_ae_sim(
            self.recording.mean_wf, 
            net,
            self.recording.peak_chans,
            spk_data, 
            [((i in self.recording.counts) and (self.recording.counts[i] > self.params["min_spikes"])) for i in range(self.recording.n_clust)],
            do_shft = self.params['ae_shft'],
            label=self.sim_view.latLbl,
            prog=self.sim_view.latProg
        )
        
        # save things
        np.save(os.path.join(self.recording.ks_dir, "automerge", "ae_sim.npy"), self.ae_sim)
        np.save(os.path.join(self.recording.ks_dir, "automerge", "spk_lat_peak.npy"), self.spk_lat_peak)
        np.save(os.path.join(self.recording.ks_dir, "automerge", "lat_mean.npy"), self.lat_mean)
        np.save(os.path.join(self.recording.ks_dir, "automerge", "spk_lab.npy"), self.spk_lab)

        self.sim_done = True
        self.sim_view.done_msg()

    def done_train(self):
        self.sim_view.close()
        self.sim_dock.close()
        self.create_sim_view()
        self.clust_view.set_sim(self.sim_view)


    def update_metrics(self, metrics):
        self.clust_view.update(metrics)


    def create_cluster_view(self):
        self.clust_view = ClusterView(self, self.recording.cluster_metrics, self.recording.load_spikes)
        self.clust_dock = _create_dock("ClusterView", self.clust_view)
        self.clust_view.set_status_pfx("clusters: ")
        self.gui.dockArea.addDock(self.clust_dock, position='top')

    def create_sim_load_view(self):
        self.sim_view = SimLoadView(self)
        self.sim_dock = _create_dock("SimilarityView", self.sim_view)
        self.gui.dockArea.addDock(self.sim_dock, 'bottom', self.clust_dock)

    def create_sim_view(self):
        self.sim_view = SimilarityView(
            self,
            self.recording.cluster_metrics,
            self.recording.load_spikes, 
            self.ae_sim,
            self.recording.template_similarity)
        
        self.sim_dock = _create_dock("SimilarityView", self.sim_view)
        self.sim_view.set_status_pfx("similar clusters: ")
        self.gui.dockArea.addDock(self.sim_dock, 'bottom', self.clust_dock)


    def create_wf_view(self):
        self.wf_view = WaveformView(self)
        self.wf_dock = _create_dock("WaveformView", self.wf_view, closable=True)
        self.gui.dockArea.addDock(self.wf_dock, position='right')

        # create menu
        self.wf_dock._menu.addAction("Set max channels", self.wf_view.set_max_chan)
        self.wf_dock._menu.addAction("Set starting channel", self.wf_view.set_ch_start)
        self.wf_dock._menu.addAction("Set ending channel", self.wf_view.set_ch_stop)

        self.wf_dock._menu.addSeparator()

        self.wf_dock._menu.addAction("Set # of rows", self.wf_view.set_num_rows)
        self.wf_dock._menu.addAction("Set # of columns", self.wf_view.set_num_cols)
        
        self.wf_dock._menu.addSeparator()

        self.wf_dock.mean_tgl = self.wf_dock._menu.addAction("Toggle mean waveforms")
        self.wf_dock.mean_tgl.setCheckable(True) 
        self.wf_dock.mean_tgl.setChecked(True)
        self.wf_dock.mean_tgl.toggled.connect(self.wf_view.toggle_mean)

        self.wf_dock.olap_tgl = self.wf_dock._menu.addAction("Toggle waveform overlap")
        self.wf_dock.olap_tgl.setCheckable(True) 
        self.wf_dock.olap_tgl.toggled.connect(self.wf_view.toggle_olap)

        self.wf_dock.lbl_tgl = self.wf_dock._menu.addAction("Toggle labels")
        self.wf_dock.lbl_tgl.setCheckable(True)
        self.wf_dock.lbl_tgl.setChecked(True)
        self.wf_dock.lbl_tgl.toggled.connect(self.wf_view.toggle_labels)


    def create_ccg_view(self):
        self.ccg_view = CorrelogramView(self)
        self.ccg_dock = _create_dock("CorrelogramView", self.ccg_view, closable=True)
        self.gui.dockArea.addDock(self.ccg_dock, 'bottom', self.wf_dock)

        # create_menu
        self.ccg_dock._menu.addAction("Set refractory period", self.ccg_view.set_ref_p)
        self.ccg_dock._menu.addAction("Set bin width", self.ccg_view.set_bin_width)
        self.ccg_dock._menu.addAction("Set window size", self.ccg_view.set_window)

        self.ccg_dock._menu.addSeparator()

        self.ccg_dock.lbl_tgl = self.ccg_dock._menu.addAction("Toggle labels")
        self.ccg_dock.lbl_tgl.setCheckable(True)
        self.ccg_dock.lbl_tgl.setChecked(True)
        self.ccg_dock.lbl_tgl.toggled.connect(self.ccg_view.toggle_labels)


    def create_fr_view(self):
        self.fr_view = FiringRateView(self)
        self.fr_dock = _create_dock("FiringRateView", self.fr_view, closable=True)
        self.gui.dockArea.addDock(self.fr_dock, "bottom", self.wf_dock)

        # create menu
        self.fr_dock._menu.addSeparator()
        self.fr_dock._menu.addAction("Set bin width", self.fr_view.set_bin_width)
        self.fr_dock._menu.addAction("Set n bins", self.fr_view.set_n_bins)
        self.fr_dock._menu.addAction("Set x min", self.fr_view.set_x_min)
        self.fr_dock._menu.addAction("Set x max", self.fr_view.set_x_max)


    def create_isi_view(self):
        self.isi_view = ISIView(self)
        self.isi_dock = _create_dock("ISIView", self.isi_view, closable=True)
        self.gui.dockArea.addDock(self.isi_dock, "below", self.fr_dock)

        # create menu
        self.isi_dock._menu.addSeparator()
        self.isi_dock._menu.addAction("Set bin width", self.isi_view.set_bin_width)
        self.isi_dock._menu.addAction("Set n bins", self.isi_view.set_n_bins)
        self.isi_dock._menu.addAction("Set x min", self.isi_view.set_x_min)
        self.isi_dock._menu.addAction("Set x max", self.isi_view.set_x_max)


    def create_prb_view(self):
        self.prb_view = ProbeView(self)
        self.prb_dock = _create_dock("ProbeView", self.prb_view, closable=True)
        self.gui.dockArea.addDock(self.prb_dock, 'right')

        # create menu
        self.prb_dock._menu.addAction("Set amplitude threshold", self.prb_view.set_amp_frac)
        self.prb_dock._menu.addAction("Set max channels", self.prb_view.set_max_chan)
        self.prb_dock.mode_tgl = self.prb_dock._menu.addAction("Toggle amplitude thresholding")
        self.prb_dock.mode_tgl.setCheckable(True)
        self.prb_dock.mode_tgl.setChecked(True)
        self.prb_dock.mode_tgl.toggled.connect(self.prb_view.toggle_mode)


    # clustering functions
    #TODO: merge mean waveforms
    #TODO: disable merge menu action before autoencoder is done training
    def merge(self):
        cl_list = self.clust_view.selected + self.sim_view.selected
        print("Merging " + str(cl_list))

        # new cluster id
        new_id = self.recording.merge(cl_list)
        new_ind = len(self.clust_view.metrics)
        cl_ind = self.clust_view.metrics['cluster_id'].isin(cl_list)

        # metrics (sum n_spikes and num_viol, avg+truncate ch and sh and peak_channel, avg other numerical, keep 1st for text, label good if all good else mua)
        # phy keeps everything from the first or largest cluster for some reason? wtfffff???????
        orig_dtype = self.clust_view.metrics.dtypes.to_dict()
        self.clust_view.metrics.loc[new_ind, :] = self.clust_view.metrics.loc[cl_ind].sum(axis=0)
        self.clust_view.metrics = self.clust_view.metrics.astype(orig_dtype)
        self.clust_view.metrics.loc[new_ind, 'cluster_id'] = new_id

        cl_ind = self.clust_view.metrics['cluster_id'].isin(cl_list) # updating to match new df length

        int_cols = self.clust_view.metrics.select_dtypes(include=np.integer).columns
        float_cols = self.clust_view.metrics.select_dtypes(include=np.inexact).columns
        string_cols = self.clust_view.metrics.select_dtypes(include='object').columns

        exclude = ['cluster_id', 'n_spikes', 'num_viol', 'fr', 'group', 'KSLabel']

        # avg + truncate int columns other than n_spikes and num_viol
        for i in range(len(int_cols)):
            if (int_cols[i] not in exclude):
                 self.clust_view.metrics.loc[new_ind, int_cols[i]] = \
                    int(self.clust_view.metrics.loc[new_ind, int_cols[i]].item()/len(cl_list))

        # avg float columns
        for i in range(len(float_cols)):
            if (float_cols[i] not in exclude):
                self.clust_view.metrics.loc[new_ind, float_cols[i]] = \
                    int(self.clust_view.metrics.loc[new_ind, float_cols[i]].item()/len(cl_list))
                
        # keep 1st for non-group text
        for i in range(len(string_cols)):
            if (string_cols[i] not in exclude):
                self.clust_view.metrics.loc[new_ind, string_cols[i]] = \
                    self.clust_view.metrics.loc[np.nonzero(cl_ind.to_numpy())[0][0], string_cols[i]]
                
        # for group, good if all good, else mua
        if (self.clust_view.metrics.loc[cl_ind, 'group'] == 'good').all():
            self.clust_view.metrics.loc[new_ind, 'group'] = 'good'
        else:
            self.clust_view.metrics.loc[new_ind, 'group'] = 'mua'

        if (self.clust_view.metrics.loc[cl_ind, 'KSLabel'] == 'good').all():
            self.clust_view.metrics.loc[new_ind, 'KSLabel'] = 'good'
        else:
            self.clust_view.metrics.loc[new_ind, 'KSLabel'] = 'mua'

        # delete old rows
        self.clust_view.metrics.drop(np.nonzero(cl_ind.to_numpy())[0], axis=0, inplace=True)

        self.clust_view.update(self.clust_view.metrics)
        self.sim_view.update(self.clust_view.metrics, [new_id])

        # select new cluster on ClusterView
        ind = -1
        for i in range(self.clust_view.tbl.rowCount()):
            if int(self.clust_view.tbl.item(i, 0).text()) == new_id:
                ind = i

        self.clust_view.tbl.selectionModel().select(self.clust_view.tbl.indexFromItem(self.clust_view.tbl.item(ind, 0)), QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        self.clust_view.tbl.verticalScrollBar().setValue(self.clust_view.tbl.rowViewportPosition(ind))

        self.clust_dock.set_status("Merged " + str(cl_list)[1:-1])


    def label_clusters(self, grp, label, cur_label=""):
        if grp == 'selected':
            cl_list = self.clust_view.selected + self.sim_view.selected
        elif grp == 'all':
            cl_list = self.clust_view.filtered
        if len(cl_list) == 0:
            return

        self.clust_view.metrics.loc[self.clust_view.metrics['cluster_id'].isin(cl_list),  'group'] = label
        
        if grp != 'all' or (grp == 'all' and label == 'good'):
            self.clust_view.metrics.loc[self.clust_view.metrics['cluster_id'].isin(cl_list),  'cur_label'] = cur_label 
        else:
            self.clust_view.metrics.loc[self.clust_view.metrics['cluster_id'].isin(cl_list),  'cur_label'] = self.clust_view.filtText

        if self.sim_done:
            self.clust_view.redraw_rows(self.sim_view.old_selection, ids=cl_list)
            self.sim_view.metrics = self.clust_view.metrics
            self.sim_view.redraw_rows(self.clust_view.old_selection, ids=cl_list)
        else:
            self.clust_view.redraw_rows(ids=cl_list)

        status = "Labelled clusters " + str(cl_list)[-1:1] + " as %s" % (label) if grp == 'selected' else "Labelled all as %s" % label
        if cur_label != "":
            status += " (%s)" % cur_label
        self.clust_dock.set_status(status)


    def sel_change(self): # if remaking graphics layout is too slow, split this into one signal for clearing and one signal for adding
        if self.sim_done:
            self.done_train()
            self.sim_done = False

        cl_list = self.clust_view.selected + self.sim_view.selected

        # replot wf view
        self.wf_view.update_wf(cl_list, self.recording.mean_wf, self.recording.spikes)

        # replot correlogram view
        self.ccg_view.update_ccg(cl_list, self.recording.cl_times)

        # replot firing rate view
        self.fr_view.update_fr(cl_list, self.recording.cl_times)

        # replot ISIview
        self.isi_view.update_isi(cl_list, self.recording.cl_times)

        # replot probeView
        self.prb_view.update_prb(cl_list)



if __name__ == '__main__':
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    control = Controller()
    sys.exit(app.exec())
