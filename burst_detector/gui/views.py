from main_window import Dock
from PyQt5.QtWidgets import (
    QAbstractItemView, QStyledItemDelegate, QTableWidgetItem,  QGraphicsPathItem, QGraphicsView, QInputDialog, QWidget,
    QVBoxLayout, QLineEdit, QProgressBar, QLabel

)
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtCore import Qt, QRectF
from pyqtgraph import TableWidget, GraphicsLayoutWidget, mkPen, mkBrush, BarGraphItem
import pyqtgraph as pg
import numpy as np
import pandas as pd
import math
import burst_detector as bd

pg.setConfigOptions(antialias=True)
#TODO: convert these to hex
colors = [(0.3528824480447752, 0.5998034969453555, 0.9971704175788023),
            (0.9832565730779054, 0.3694984452949815, 0.3488265255379734),
            (0.4666666666666667, 0.8666666666666667, 0.4666666666666667),
            (0.999, 0.8666666666666666, 0.23116059580240228),
            (0.999, 0.62745156, 0.3019607),
            (0.656421832660253, 0.35642078793464527, 0.639125729774389),
            (0.999, 0.6509803921568628, 0.788235294117647),
            (0.8352941176470589, 0.6392156862745098, 0.4470588235294118),
            (0.25098039215686274, 0.8784313725490196, 0.8156862745098039),
            (0.7098039215686275, 0.7411764705882353, 0.6470588235294118)]

light_colors = [(229/255,239/255,254/255),
            (254/255,231/255,230/255),
            (234/255,249/255,234/255),
            (254/255,250/255,229/255),
            (254/255,241/255,229/255),
            (246/255,238/255,245/255),
            (254/255,229/255,239/255),
            (249/255,242/255,235/255),
            (233/255,251/255,249/255),
            (243/255,243/255,240/255)]


# TODO: pending for cluster/similarity view
# - splitting

class CustomItemDelegate(QStyledItemDelegate):
    """
    Custom painter to handle Cluster/SimilarityView coloring
    """
    colors = [(0.3528824480447752, 0.5998034969453555, 0.9971704175788023),
            (0.9832565730779054, 0.3694984452949815, 0.3488265255379734),
            (0.4666666666666667, 0.8666666666666667, 0.4666666666666667),
            (0.999, 0.8666666666666666, 0.23116059580240228),
            (0.999, 0.62745156, 0.3019607),
            (0.656421832660253, 0.35642078793464527, 0.639125729774389),
            (0.999, 0.6509803921568628, 0.788235294117647),
            (0.8352941176470589, 0.6392156862745098, 0.4470588235294118),
            (0.25098039215686274, 0.8784313725490196, 0.8156862745098039),
            (0.7098039215686275, 0.7411764705882353, 0.6470588235294118)]

    light_colors = [(229/255,239/255,254/255),
                (254/255,231/255,230/255),
                (234/255,249/255,234/255),
                (254/255,250/255,229/255),
                (254/255,241/255,229/255),
                (246/255,238/255,245/255),
                (254/255,229/255,239/255),
                (249/255,242/255,235/255),
                (233/255,251/255,249/255),
                (243/255,243/255,240/255)]
    
    # TODO: Pick better colors 
    group_colors = {'good': 0x86D16D, 'mua': 0x616161, 'noise': 0x909090}

    def __init__(self, table):
        super().__init__()
        self.table = table  # reference to parent TableWidget

    def paint(self, painter, option, index):
        # convert table row index to metrics DataFrame row index
        row_ind = self.table.par.metrics["cluster_id"] ==  int(self.table.item(index.row(), 0).text())
        label = self.table.par.metrics.loc[row_ind, 'group'].item()

        # set text color based on cluster_group
        txt_color = self.group_colors[label] if label in self.group_colors else 0x000000
        option.palette.setColor(QPalette.Text, QColor(txt_color))
  
        # set highlight/text styles for selected rows, cluster_id column has different style from the rest
        for i in range(len(self.table.par.selected)):
            # check if the item being painted is part of a selected row
            if int(self.table.item(index.row(), 0).text()) == self.table.par.selected[i]: 
                cind = (i+self.table.par.offset)%10
                if index.column() == 0: # item is a cluster_id entry
                    color = QColor(int(255*self.colors[cind][0]), int(255*self.colors[cind][1]), int(255*self.colors[cind][2]))
                    option.palette.setColor(QPalette.Highlight, color)
                    option.font.setBold(True)
                else:                   # item is not a cluster_id entry
                    color = QColor(int(255*self.colors[cind][0]), int(255*self.colors[cind][1]), int(255*self.colors[cind][2]), 50)
                    option.palette.setColor(QPalette.Highlight, color)
                    option.palette.setColor(QPalette.HighlightedText, QColor(txt_color))
        super().paint(painter, option, index)


class ClusterView(QWidget):
    def __init__(self, controller, metrics, load_spikes):
        super().__init__()

        # init filtering text entry box
        self.le = QLineEdit(parent=self)
        self.le.setStyleSheet("QLineEdit {border-style: outset; border-width: 2px; border-color: black; \
                              background-color: black; color: white;}")
        self.le.setPlaceholderText("filter")
        self.le.editingFinished.connect(self.filter)
        self.le.textChanged.connect(self.checkFilterReset)
        self.filtText = ""

        # init TableWidget
        self.tbl = TableWidget(0,0)
        self.tbl.par = self
        self.tbl.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.itemSelectionChanged.connect(self.on_selection_change)
        self.tbl.horizontalHeader().sectionClicked.connect(self.on_sort_change)

        # TableWidget style
        p = self.tbl.palette()
        p.setColor(QPalette.Base, Qt.black)
        self.tbl.setPalette(p)
        self.tbl.setItemDelegate(CustomItemDelegate(table=self.tbl))
        self.tbl.setShowGrid(False)
        # self.tbl.horizontalHeader().setStyleSheet("QHeaderView::section {background-color: black; color:white;} \
        #                                      QHeaderView::section:selected {background-color: black; color:white;}")

        # add widgets to view
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.addWidget(self.le)
        self.layout.addWidget(self.tbl)
        self.setLayout(self.layout)

        # populate table
        self.sort_col = 0
        self.sort_order = Qt.AscendingOrder
        self.update(metrics)
        for i in range(self.metrics.shape[0]):
            self.tbl.setRowHeight(i, 20)

        # init state variables
        self.controller = controller
        self.load_spikes = load_spikes

        self.filtered = self.metrics['cluster_id'].tolist()
        self.selected = []
        self.old_selection = []
        self.offset = 0

        self.sim = None

    def on_selection_change(self):
        selected_ids = [int(self.tbl.item(index.row(), 0).text()) for index in self.tbl.selectionModel().selectedRows()] 

        selected = [id for id in self.selected if id in selected_ids]       # remove deselected
        new = [id for id in selected_ids if id not in self.selected]        # add new selections    

        # cache spikes for selected rows
        self.selected = selected + new
        for id in self.selected:
            self.load_spikes(id)

        self.redraw_rows()
        # notify SimilarityView of selection change
        if self.sim is not None:
            self.sim.on_clust_sel_change(self.selected)

        # update status bar
        if len(self.selected) > 0:
            self.dock.set_status(self.status_pfx + str(self.selected)[1:-1])
        else:
            self.dock.set_status("")

        # notify controller of selection change
        self.controller.sel_change()

    def setDock(self, dock):
        self.dock = dock

    def set_status_pfx(self, pfx):
        self.status_pfx = pfx

    def redraw_rows(self, extra_rows=[], ids=[]):
        # re-paint rows referenced by table index -- previously selected, currently selected, any extra rows
        for index in self.old_selection:
            item = self.tbl.item(index.row(), self.tbl.columnCount()-1)
            self.tbl.model().dataChanged.emit(index, self.tbl.indexFromItem(item))
        for index in self.tbl.selectionModel().selectedRows():
            item = self.tbl.item(index.row(), self.tbl.columnCount()-1)
            self.tbl.model().dataChanged.emit(index, self.tbl.indexFromItem(item))
        for index in extra_rows:
            item = self.tbl.item(index.row(), self.tbl.columnCount()-1)
            self.tbl.model().dataChanged.emit(index, self.tbl.indexFromItem(item))

        # re-paint rows referenced by cluster id (currently only used to add curation label)
        if len(ids) != 0:
            for i in range(self.tbl.rowCount()):
                item = self.tbl.item(i, 0)
                if int(item.text()) in ids:
                    col_ind = 4 if self.sim is not None else 6
                    self.tbl.setItem(i, col_ind, QTableWidgetItem(
                        self.metrics.loc[self.metrics['cluster_id'] == int(item.text()), 'cur_label'].item()))
                    end = self.tbl.item(i, self.tbl.columnCount()-1)
                    self.tbl.model().dataChanged.emit(self.tbl.indexFromItem(item), self.tbl.indexFromItem(end))

        # update memory of selection
        self.old_selection = self.tbl.selectionModel().selectedRows()

    def set_sim(self, sim_view):
        self.sim = sim_view

    def update(self, metrics):
        self.le.clear()

        # set new data, adjust table parameters accordingly
        self.tbl.setRowCount(metrics.shape[0])
        self.tbl.setColumnCount(metrics.shape[1]) 
        self.metrics = metrics
        self.tbl.setData(metrics.drop('group', axis=1).to_numpy())
        self.tbl.setHorizontalHeaderLabels(self.metrics.drop('group', axis=1).columns)
 
        # style
        self.tbl.resizeColumnsToContents()
        self.tbl.horizontalHeader().setStyleSheet("font-weight: bold;")
        self.tbl.horizontalHeader().setFirstSectionMovable(False)
        self.tbl.horizontalHeader().setSectionsMovable(True)
        self.tbl.verticalHeader().setVisible(False)

        self.tbl.sortItems(self.sort_col, self.sort_order)

    def on_sort_change(self, col):
        if self.sort_col == col: # switch direction if already sorting by this column 
            self.sort_order = Qt.DescendingOrder if self.sort_order == Qt.AscendingOrder else Qt.AscendingOrder
        else:
            self.sort_col = col # ascending if new column
            self.sort_order = Qt.AscendingOrder

    def filter(self):
        try:
            ids = self.metrics.query(self.le.text())['cluster_id'].tolist()
            self.filtText = self.le.text()
            self.filtered = ids
            self.hide_rows()
        except (SyntaxError, ValueError, pd.core.computation.ops.UndefinedVariableError):
            return
        
    def checkFilterReset(self, text):
        if text == "":
            self.filtered = self.metrics['cluster_id'].tolist()
            self.hide_rows()
    
    def hide_rows(self):
        for i in range(self.tbl.rowCount()):
            if int(self.tbl.item(i, 0).text()) not in self.filtered:    # hide rows that don't meet filter boolean
                self.tbl.hideRow(i)
            # for SimilarityView, also hide rows that are selected in ClusterView
            elif ('ae_sim' in self.metrics) and (int(self.tbl.item(i, 0).text()) in self.selected):
                self.tbl.hideRow(i)
            else:
                self.tbl.showRow(i)

class SimilarityView(ClusterView):
    def __init__(self, controller, metrics, load_spikes, ae_sim, temp_sim):
        super().__init__(controller, metrics, load_spikes)
        self.set_sim(None)

        # start empty
        for i in range(self.tbl.rowCount()):
            self.tbl.hideRow(i)

        # adjust table parameters
        col_names = self.metrics.drop('group', axis=1).columns.values.tolist()
        col_names.insert(1, 'tmp_sim')
        col_names.insert(1, 'ae_sim')
        self.metrics.insert(1, 'tmp_sim', np.zeros(self.metrics.shape[0]))
        self.metrics.insert(1, 'ae_sim', np.zeros(self.metrics.shape[0]))
        self.tbl.setHorizontalHeaderLabels(col_names)
        self.tbl.resizeColumnsToContents()

        # similarity view state variables
        self.ae_sim = ae_sim
        self.temp_sim = temp_sim
        self.sort_col = 1
        self.sort_order = Qt.DescendingOrder    
        self.tbl.sortItems(1, Qt.DescendingOrder)

        self.tbl.setItemDelegate(CustomItemDelegate(table=self.tbl))

    def on_clust_sel_change(self, cv_selected):
        # update state variables, clear SimilarityView selection
        self.cv_selected = cv_selected   
        self.selected = []
        self.offset = len(cv_selected)
        self.redraw_rows()
        self.tbl.clearSelection()

        # hide rows that are selected in ClusterView
        for i in range(self.tbl.rowCount()):
            if int(self.tbl.item(i, 0).text()) in cv_selected:
                self.tbl.hideRow(i)
            elif int(self.tbl.item(i, 0).text()) not in self.filtered:
                self.tbl.hideRow(i)
            else:
                self.tbl.showRow(i)

        # populate similarity with temp_sim for earliest selected cluster
        if len(cv_selected):
            self.metrics['ae_sim'] = self.ae_sim[cv_selected[0], self.metrics['cluster_id'].astype("int")]
            self.metrics['tmp_sim'] = self.temp_sim(cv_selected[0])[self.metrics['cluster_id'].astype("int")]
            self.tbl.setSortingEnabled(False)
            
            for i in range(self.tbl.rowCount()):
                clust = int(self.tbl.item(i, 0).text())
                ae_sim = self.metrics.loc[self.metrics['cluster_id'] == clust, 'ae_sim'].item()
                tmp_sim = self.metrics.loc[self.metrics['cluster_id'] == clust, 'tmp_sim'].item()
                self.tbl.setItem(i, 1, QTableWidgetItem(f"{ae_sim:.3f}"))
                self.tbl.setItem(i, 2, QTableWidgetItem(f"{tmp_sim:.3f}"))

            self.tbl.setSortingEnabled(True)

    # write custom update function that keeps similarity column
    def update(self, metrics, cv_selected=None):
        self.le.clear()

        # set new data, adjust table parameters accordingly
        self.tbl.setRowCount(metrics.shape[0])
        self.tbl.setColumnCount(metrics.shape[1]) 
        self.tbl.setData(metrics.drop('group', axis=1).to_numpy())
        self.metrics = metrics

        # add similarity columns
        self.tbl.insertColumn(1)
        self.tbl.insertColumn(1)
        col_names = self.metrics.drop('group', axis=1).columns.values.tolist()
        col_names.insert(1, 'tmp_sim')
        col_names.insert(1, 'ae_sim')
        self.tbl.setHorizontalHeaderLabels(col_names)
 
        # style
        self.tbl.resizeColumnsToContents()
        self.tbl.horizontalHeader().setStyleSheet("font-weight: bold;")
        self.tbl.horizontalHeader().setFirstSectionMovable(False)
        self.tbl.horizontalHeader().setSectionsMovable(True)
        self.tbl.verticalHeader().setVisible(False)

        self.tbl.sortItems(self.sort_col, self.sort_order)

class SimLoadView(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.ctl = controller
        self.selected = []

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(0)

        self.extractLbl = QLabel(self)
        self.trainLbl = QLabel(self)
        self.latLbl = QLabel(self)

        self.extractProg = QProgressBar(self)
        self.extractProg.setTextVisible(True)
        self.trainProg = QProgressBar(self)
        self.trainProg.setTextVisible(True)
        self.latProg = QProgressBar(self)
        self.latProg.setTextVisible(True)

        self.layout.addWidget(self.extractLbl)
        self.layout.addWidget(self.extractProg)
        self.layout.addStretch()
        self.layout.addWidget(self.trainLbl)
        self.layout.addWidget(self.trainProg)
        self.layout.addStretch()
        self.layout.addWidget(self.latLbl)
        self.layout.addWidget(self.latProg)

        self.setLayout(self.layout)

        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)
        self.setAutoFillBackground(True)

        p = self.extractProg.palette()
        p.setColor(QPalette.WindowText, Qt.white)
        self.extractProg.setPalette(p)
        self.trainProg.setPalette(p)
        self.latProg.setPalette(p)

        self.extractLbl.setWordWrap(True)
        self.trainLbl.setWordWrap(True)
        self.latLbl.setWordWrap(True)

        p = self.extractLbl.palette()
        p.setColor(QPalette.WindowText, Qt.white)
        self.extractLbl.setPalette(p)
        self.trainLbl.setPalette(p)
        self.latLbl.setPalette(p)

        self.done_lbl = QLabel("")

        p = self.done_lbl.palette()
        p.setColor(QPalette.WindowText, Qt.white)
        self.done_lbl.setPalette(p)
        self.done_lbl.setWordWrap(True)

        self.layout.addStretch()
        self.layout.addWidget(self.done_lbl)
        

    def setDock(self, dock):
        self.dock = dock

    def done_msg(self):
        self.done_lbl.setText("Done with autoencoder! Select a different cluster to show SimilarityView.")

# base class for all plot-related views
class GraphView(GraphicsLayoutWidget):
    def __init__(self, controller, **kwargs):
        super().__init__(border=(0,0,0))
        self.ctl = controller
        self.ci.layout.setContentsMargins(0,0,0,0)
        self.ci.layout.setSpacing(0)

        # trackers for pan/zoom
        self.lr = 0
        self.ud = 0
        self.zoom = 0

        self.scaleCenter=True

    def setDock(self, dock):
        self.dock = dock

    # panning and zooming using keyboard
    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Left:
            QGraphicsView.translate(self, float(self.pixelSize().x()*20), 0)
            self.lr -= 1
        elif ev.key() == Qt.Key_Right:
            QGraphicsView.translate(self, -1*float(self.pixelSize().x()*20), 0)
            self.lr += 1
        elif ev.key() == Qt.Key_Up:
            QGraphicsView.translate(self, 0, float(self.pixelSize().y()*20))
            self.ud -= 1
        elif ev.key() == Qt.Key_Down:
            QGraphicsView.translate(self, 0, -1*float(self.pixelSize().y()*20))
            self.ud += 1
        elif ev.key() == Qt.Key_Minus:
            QGraphicsView.scale(self, .95, .95)
            self.zoom -= 1
        elif ev.key() == Qt.Key_Equal:
            QGraphicsView.scale(self, 1/0.95, 1/0.95)
            self.zoom += 1
        elif ev.key() == Qt.Key_R:
            QGraphicsView.scale(self, .95**(self.zoom), .95**(self.zoom))
            QGraphicsView.translate(self, -1*self.lr*self.pixelSize().x()*20, -1*self.ud*self.pixelSize().y()*20)
            self.lr = 0
            self.ud = 0
            self.zoom = 0

        self.scene().keyPressEvent(ev)  ## bypass view, hand event directly to scene
                                        ## (view likes to eat arrow key events)   


# TODO: change wf plot order to match probe geometry
# TODO: make amplitude magnitude obvious somehow -- maybe plot ch. std deviation background or plot all wf relative to max amp?
# TODO: MERGE ME WITH PROBE VIEW
class WaveformView(GraphView):
    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)

        # channel range state variables
        self.ch_start = -1
        self.ch_stop = -1
        self.ch_max = 10

        # channel grid size state variables
        self.num_cols = -1
        self.num_rows = -1

        # togglable variables
        self.wf_mode = 'mean'
        self.olap = False
        self.show_labels = True
        self.amp_thresh = True
        self.amp_frac = 1/3

        # current plots state variables
        self.cl_list = None
        # self.ch_lo = {}
        self.plots = []

        self.ch_grid = None
        self.cl_grid = None
        self.ch_lo = None        # num cols x num rows
        self.items = None        # num cols x num rows x num clusters

    def update_wf(self, cl_list, mean_wf, spikes):
        # # calculate plot layout
        # ch_grid, cl_grid, ch_range, status = self.calc_layout(cl_list, mean_wf)
        # if status == "mid_update":
        #     return

        # # if (ch_grid == self.ch_grid) and (cl_grid == self.cl_grid):
        # #     return
        
        # # first time plotting
        # if self.ch_grid == None:
        #     ch_lo = np.zeros_like(ch_grid, dtype='object')
        #     items = np.zeros((ch_lo.shape[0], ch_lo.shape[1], len(cl_list)))

        #     # calculate ylim
        #     dat_min = []
        #     dat_max = []
        #     for cl in cl_list:
        #         dat_min.append((mean_wf[cl,ch_range[0]:ch_range[1],:]).min()-10)
        #         dat_max.append((mean_wf[cl,ch_range[0]:ch_range[1],:]).max()+10)
        #     dat_min = min(dat_min)
        #     dat_max = max(dat_max)
        #     ymin = 2*dat_min if dat_min < 0 else dat_min - .3 * (dat_max-dat_min)
        #     ymax = 2*dat_max if dat_max > 0 else dat_max + .3 * (dat_max-dat_min)

        #     for i in range(ch_grid.shape[0]):
        #         for j in range(ch_grid.shape[1]):
        #             # create a layout for each channel
        #             ch_lo[i,j] = self.addLayout(row=i, col=j)
        #             ch = ch_grid[i,j]

        #             # create plots
        #             for k in range(cl_grid.shape[2]):
        #                 if self.show_labels:
        #                     ch_lo[i,j].addLabel(str(ch), angle=0, color="#FFFFFF", size="10pt")
        #                 plot = ch_lo[i,j].addPlot(row=0, col=k+1)
        #                 plot.vb.sigRangeChanged.connect(self.update_range)
        #                 self.plots.append(plot)

        #             # do plotting
        #             ind = 0
        #             for k in range(cl_grid.shape[2]):
        #                 plot = ch_lo[i,j].getItem(row=0, col=k+1)
        #                 clusts = cl_grid[i,j,k]

        #                 for cl in clusts:    
        #                     color = (int(colors[ind][0]*255), int(colors[ind][1]*255), int(colors[ind][2]*255))
        #                     if self.wf_mode == 'mean':
        #                         items[i,j,ind] = self.draw_mean(plot, mean_wf[cl, ch, :40], color, ymin, ymax)
        #                     else:
        #                         items[i,j,ind] = self.draw_spikes(plot, spikes[cl][:,ch,:40], color, ymin, ymax)
                        
        #                     ind += 1

        #     # update state variables
        #     self.items = items
        #     self.ch_grid = ch_grid
        #     self.cl_grid = cl_grid
        #     self.ch_lo = ch_lo
        #     self.ch_start = ch_range[0]
        #     self.ch_stop = ch_range[1]

        #     return

        # # modifying existing plots

        # # remove or add channel layout as needed

        # # remove or add plots to channel layouts as needed
        
        # # #TODO: speed optimizations? speed is good for spikes, 
        # # # but should distinguish between adding clusters vs reset to make visualizing many clusters faster

        # do nothing if we're in the middle of updating view options
        if (self.ch_stop != -1) and (self.ch_start > self.ch_stop): # we're in the middle of changing channel bounds
            return 
        if (self.ch_stop > self.ch_start+self.ch_max):
            return 
        if not cl_list: # we're in the middle of merging
            return  
        
        # reset channel range if cluster selection is different
        if (self.cl_list is not None) and (self.cl_list[0] != cl_list[0]):
            self.ch_start = -1
            self.ch_stop = -1

        # clear plot, update selected
        self.clear()
        self.cl_list = cl_list
        self.ch_lo = {}
        self.plots = []

        # center channel range around average peak if not specified
        if (self.ch_start == -1) and (self.ch_stop == -1):
            peaks = []
            for cl in cl_list:
                peaks.append(np.argmax(np.max(mean_wf[cl], 1) - np.min(mean_wf[cl], 1)))
            peak = sum(peaks)/len(cl_list)
            ch_start = max(int(peak-self.ch_max/2+1), 0)
            ch_stop = min(int(peak+self.ch_max/2), mean_wf.shape[1]-1)

            # center around first selected cluster if peak doesn't include first peak
            if peaks[0] < ch_start or peaks[0] > ch_stop:
                peak = peaks[0]
                ch_start = max(int(peak-self.ch_max/2+1), 0)
                ch_stop = min(int(peak+self.ch_max/2), mean_wf.shape[1]-1)
        # otherwise set relative to the specified bound
        elif self.ch_start == -1:
            self.ch_start = max(0,self.ch_stop - self.ch_max + 1)
        elif self.ch_stop == -1:
            self.ch_stop = min(self.ctl.recording.params['n_channels_dat'], self.ch_start + self.ch_max - 1)
        else:
            ch_stop = self.ch_stop
            ch_start = self.ch_start

        # calculate grid size
        if (self.num_rows == -1) and (self.num_cols == -1):      # default number of columns if both are -1
            num_cols = 3
        if (self.num_rows == -1):
            num_cols = self.num_cols if (self.num_cols != -1) else num_cols
            num_rows = math.ceil((ch_stop-ch_start+1)/num_cols)
        elif (self.num_cols == -1):
            num_rows = self.num_rows if (self.num_rows != -1) else num_rows
            num_cols = math.ceil((ch_stop-ch_start+1)/num_rows)
        else:
            num_rows = self.num_rows
            num_cols = self.num_cols

        # calculate ylim
        dat_min = []
        dat_max = []
        for cl in cl_list:
            dat_min.append((mean_wf[cl,ch_start:ch_stop,:]).min()-10)
            dat_max.append((mean_wf[cl,ch_start:ch_stop,:]).max()+10)
        dat_min = min(dat_min)
        dat_max = max(dat_max)

        ymin = 2*dat_min if dat_min < 0 else dat_min - .3 * (dat_max-dat_min)
        ymax = 2*dat_max if dat_max > 0 else dat_max + .3 * (dat_max-dat_min)

        # plot
        for i in range(ch_start, ch_stop+1):
            # add one layout per channel
            self.ch_lo[i] = self.addLayout(row=(i-ch_start)%num_rows, col=int((i-ch_start)/num_rows))  
            if self.show_labels:
                self.ch_lo[i].addLabel(str(i), angle=0, color="#FFFFFF", size="10pt")

            # plot each cluster
            for j in range(len(cl_list)):
                cl = cl_list[j]
                # add a new plot if don't want overlapping or plotting a new channel
                if (not self.olap) or (self.olap and (j == 0)):
                    plot = self.ch_lo[i].addPlot()
                    plot.vb.sigRangeChanged.connect(self.update_range)

                    self.plots.append(plot)
                    plot.vb.setYRange(ymin, ymax)

                else:
                    plot = self.plots[i-ch_start]

                color = (int(colors[j][0]*255), int(colors[j][1]*255), int(colors[j][2]*255))
                # draw waveforms
                if self.wf_mode == 'mean':
                    self.draw_mean(plot, mean_wf[cl, i, :40], color, ymin, ymax)
                else:
                    self.draw_spikes(plot, spikes[cl][:,i,:40], color, ymin, ymax)

        # update state variables
        self.ch_start = ch_start
        self.ch_stop = ch_stop

        status = 'mean waveforms' if self.wf_mode == 'mean' else 'spike waveforms'
        self.dock.set_status(status)

    def calc_layout(self, cl_list, mean_wf):
        # skip update if necessary
        # do nothing if we're in the middle of updating view options
        if (self.ch_stop != -1) and (self.ch_start > self.ch_stop): # we're in the middle of changing channel bounds
            return None, 'mid_update'
        if (self.ch_stop > self.ch_start+self.ch_max):
            return None, 'mid_update'
        if not cl_list: # we're in the middle of merging
            return None, 'mid_update' 
        
        # reset channel range if cluster selection is different
        if (self.cl_list is not None) and (self.cl_list[0] != cl_list[0]):
            self.ch_start = -1
            self.ch_stop = -1

        # center channel range around average peak if not specified
        if (self.ch_start == -1) and (self.ch_stop == -1):
            peaks = []
            for cl in cl_list:
                peaks.append(np.argmax(np.max(mean_wf[cl], 1) - np.min(mean_wf[cl], 1)))
            peak = sum(peaks)/len(cl_list)
            ch_start = max(int(peak-self.ch_max/2+1), 0)
            ch_stop = min(int(peak+self.ch_max/2), mean_wf.shape[1]-1)

            # center around first selected cluster if peak doesn't include first peak
            if peaks[0] < ch_start or peaks[0] > ch_stop:
                peak = peaks[0]
                ch_start = max(int(peak-self.ch_max/2+1), 0)
                ch_stop = min(int(peak+self.ch_max/2), mean_wf.shape[1]-1)
        # otherwise set relative to the specified bound
        elif self.ch_start == -1:
            self.ch_start = max(0,self.ch_stop - self.ch_max + 1)
        elif self.ch_stop == -1:
            self.ch_stop = min(self.ctl.recording.params['n_channels_dat'], self.ch_start + self.ch_max - 1)
        else:
            ch_stop = self.ch_stop
            ch_start = self.ch_start

        # calculate grid size
        if (self.num_rows == -1) and (self.num_cols == -1):      # default number of columns if both are -1
            num_cols = 3
        if (self.num_rows == -1):
            num_cols = self.num_cols if (self.num_cols != -1) else num_cols
            num_rows = math.ceil((ch_stop-ch_start+1)/num_cols)
        elif (self.num_cols == -1):
            num_rows = self.num_rows if (self.num_rows != -1) else num_rows
            num_cols = math.ceil((ch_stop-ch_start+1)/num_rows)
        else:
            num_rows = self.num_rows
            num_cols = self.num_cols

        # channel grid
        ch_grid = np.zeros((num_rows, num_cols), dtype='int')
        for i in range(ch_start, ch_stop+1):
            row = (i-ch_start)%num_rows
            col = int((i-ch_start)/num_rows)
            ch_grid[row, col] = i

        # make layout
        if not self.olap:
            cl_grid = np.zeros((num_rows, num_cols, len(cl_list)), dtype="object")

            for i in range(ch_start, ch_stop+1):
                for j in range(len(cl_list)):
                    row = (i-ch_start)%num_rows
                    col = int((i-ch_start)/num_rows)
                    cl_grid[row, col, j] = [cl_list[j]]

        else:
            cl_grid = np.zeros((num_rows, num_cols,1), dtype="object")
            for i in range(ch_start, ch_stop+1):
                row = (i-ch_start)%num_rows
                col = int((i-ch_start)/num_rows)
                cl_grid[row, col, 0] = cl_list

        status = 'replot'

        return ch_grid, cl_grid, [ch_start, ch_stop], status

    def update_range(self, viewBox, viewRange):
        """
        Links together the axes of plots for joint panning/zooming
        """
        for plot in self.plots:
            if plot.vb is not viewBox:
                plot.vb.blockSignals(True)  # prevent this update from triggering other updates
                plot.setRange(xRange=tuple(viewRange[0]), yRange=tuple(viewRange[1]), padding=0)
                plot.vb.blockSignals(False)

    def draw_spikes(self, plot, spikes, color, ymin, ymax):
        # initalize x-axis array
        x = np.empty((spikes.shape[0], spikes.shape[1]))
        x[:] = np.arange(spikes.shape[1])

        # we plot one big disconnected line to avoid instantiating many individual PlotItems
        # connect tells the painter which data points should be connected
        connect = np.ones((spikes.shape[0], spikes.shape[1]), dtype=np.ubyte)
        connect[:,-1] = 0  #  disconnect segment between lines

        tot_samp = spikes.shape[0] * spikes.shape[1]
        # adding very small random noise to traces to fill in gaps from integer quantization (we have more spikes
        # than unique values at each timepoint, causes diamond-shaped gaps to appear in the plot)
        path = pg.arrayToQPath(x.reshape(tot_samp), spikes.reshape(tot_samp)+.3*np.random.randn(tot_samp), connect.reshape(tot_samp))
        item = QGraphicsPathItem(path)

        # plot
        pen = mkPen(QColor(color[0], color[1], color[2]), width=.4)
        item.setPen(pen)
        plot.addItem(item)
       
        # axis view options
        vb = plot.getViewBox()
        vb.setYRange(ymin, ymax)
        plot.hideAxis('left'); plot.hideAxis('bottom'); plot.hideButtons()

    def draw_mean(self, plot, data, color, ymin, ymax):
        # plot
        pen = mkPen(QColor(color[0], color[1], color[2]), width=2)
        plot.plot(data, pen=pen)

        # axis view options
        vb = plot.getViewBox()
        vb.setYRange(ymin, ymax)
        plot.hideAxis('left'); plot.hideAxis('bottom'); plot.hideButtons()


    ## OPTION FUNCTIONS
    def set_max_chan(self):
        chan, ok = QInputDialog.getInt(self, 'Set max channels', 'Set the maximum number of channels to be displayed (max 20).', 
                                       value=self.ch_max, min=1, max=20, step=1)
        if ok and self.ch_max != chan:
            self.ch_max = chan
            self.update_wf(self.cl_list, self.ctl.recording.mean_wf, self.ctl.recording.spikes)

    def set_ch_start(self):
        ch_start, ok = QInputDialog.getInt(self, 'Set starting channel', 'Set the first channel shown in the WaveformView',
                                           value = self.ch_start, min=0, 
                                           max= self.ctl.recording.params['n_channels_dat']-1, step=1)
        if ok and self.ch_start != ch_start:
            self.ch_start = ch_start
            self.update_wf(self.cl_list, self.ctl.recording.mean_wf, self.ctl.recording.spikes)

    def set_ch_stop(self):
        ch_stop, ok = QInputDialog.getInt(self, 'Set ending channel', 'Set the last channel shown in the WaveformView',
                                           value = self.ch_stop, min=0, 
                                           max=  self.ctl.recording.params['n_channels_dat']-1,
                                           step=1)
        if ok and self.ch_stop != ch_stop:
            self.ch_stop = ch_stop
            self.update_wf(self.cl_list, self.ctl.recording.mean_wf, self.ctl.recording.spikes)

    # TODO: replace grid plotting functions with something that doesn't replot everything
    def set_num_rows(self):
        num_rows, ok = QInputDialog.getInt(self, 'Set number of rows', 'Set the number of channel rows shown in the WaveformView. \
                                           \n(set -1 to set rows automatically based on number of columns)', 
                                           value = self.num_rows, min = -1, max = self.ch_max, step=1)
        if ok and self.num_rows != num_rows and num_rows != 0:
            self.num_rows = num_rows
            self.update_wf(self.cl_list, self.ctl.recording.mean_wf, self.ctl.recording.spikes)

    def set_num_cols(self):
        num_cols, ok = QInputDialog.getInt(self, 'Set number of columns', 'Set the number of channel columns shown in the WaveformView. \
                                           \n(set -1 to set columns automatically based on number of rows)', 
                                           value = self.num_cols, min = -1, max = self.ch_max, step=1)
        if ok and self.num_cols != num_cols and num_cols != 0:
            self.num_cols = num_cols
            self.update_wf(self.cl_list, self.ctl.recording.mean_wf, self.ctl.recording.spikes)

    def toggle_mean(self, plt_mean):
        self.wf_mode = 'mean' if plt_mean else 'spike'
        # TODO: replace me with a function that clears individual plots and replots
        self.update_wf(self.cl_list, self.ctl.recording.mean_wf, self.ctl.recording.spikes)

    def toggle_olap(self, olap):
        self.olap = True if olap else False
        self.update_wf(self.cl_list, self.ctl.recording.mean_wf, self.ctl.recording.spikes)

    def toggle_labels(self, label):
        self.show_labels = True if label else False
        #TODO:replace me with a function that doesn't replot everything
        self.update_wf(self.cl_list, self.ctl.recording.mean_wf, self.ctl.recording.spikes) 


# TODO: draw refractory period lines
# TODO: turn off axes automatically for > 3? clusters
class CorrelogramView(GraphView):
    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)

        # correlogram parameters
        self.ref_p = 0.002
        self.bin_width = 0.001
        self.window_size = 0.1

        self.show_labels = True
        self.acg_cache = {}

    def update_ccg(self, cl_list, cl_times):
        self.clear()
        self.cl_list = cl_list

        self.plots = np.zeros((len(cl_list), len(cl_list)), dtype=object)
        n_clust = len(cl_list)

        # compute acg
        acgs = []
        for i in range(n_clust):
            if cl_list[i] in self.acg_cache:
                acg = self.acg_cache[cl_list[i]]
            else:
                acg = bd.auto_correlogram(
                    cl_times[cl_list[i]]/float(self.ctl.recording.params['sample_rate']), 
                    window_size=self.window_size, 
                    bin_width=self.bin_width,
                    overlap_tol=self.ctl.params['overlap_tol'])
                self.acg_cache[cl_list[i]] = acg
            acgs.append(acg)

        # compute ccg
        ccgs = []
        for i in range(n_clust):
            ccgs.append([])
            for j in range(n_clust):
                if i >= j: 
                    ccgs[i].append([])
                else:
                    times_i = cl_times[cl_list[i]]/float(self.ctl.recording.params['sample_rate'])
                    times_j = cl_times[cl_list[j]]/float(self.ctl.recording.params['sample_rate'])
                    ccg = bd.x_correlogram(
                        times_i, 
                        times_j, 
                        window_size=self.window_size, 
                        bin_width=self.bin_width,
                        overlap_tol=self.ctl.params['overlap_tol'])[0]

                    ccgs[i].append(ccg)

        # plot
        for i in range(n_clust):
            color = (int(colors[i][0]*255), int(colors[i][1]*255), int(colors[i][2]*255))
            acg_pen = mkPen(QColor(color[0], color[1], color[2]), width=0)
            acg_brush = mkBrush(QColor(color[0], color[1], color[2]))

            lt_color = (int(light_colors[i][0]*255), int(light_colors[i][1]*255), int(light_colors[i][2]*255))
            ccg_pen = mkPen(QColor(lt_color[0], lt_color[1], lt_color[2]), width=0)
            ccg_brush = mkBrush(QColor(lt_color[0], lt_color[1], lt_color[2]))

            for j in range(n_clust):
                if j == 0 and self.show_labels:
                    self.addLabel(str(cl_list[i]), angle=0, color="#FFFFFF", size="10pt", row=i, col=0)
                self.plots[i,j] = self.addPlot(row=i, col=j+1)
    
                # self.plots[i,j].vb.sigRangeChanged.connect(self.update_range)
                if i != j:
                    ccg = ccgs[i][j] if i < j else ccgs[j][i][::-1]
                    if ccg.any():
                        x = np.arange(ccg.shape[0])
                        self.plots[i,j].addItem(BarGraphItem(x=x, height=ccg, width=1, pen=ccg_pen, brush=ccg_brush))

                if i == j:
                    # plot acg
                    if acgs[i].any():
                        x = np.arange(acgs[i].shape[0])
                        self.plots[i,j].addItem(BarGraphItem(x=x, height=acgs[i], width=1, pen=acg_pen, brush=acg_brush))
                if (i==n_clust-1) and self.show_labels:
                    self.addLabel(str(cl_list[j]), angle=0, color="#FFFFFF", size="10pt", row=n_clust, col=j+1)

        self.dock.set_status("window: %.1f ms, bin: %.1f ms" % (self.window_size*1000, self.bin_width*1000))

    def update_range(self, viewBox, viewRange):
        for i in range(self.plots.shape[0]):
            for j in range(self.plots.shape[0]):
                plot = self.plots[i,j]
                if plot and plot.vb is not viewBox:
                    # plot.vb.blockSignals(True)  # prevent this update from triggering other updates
                    plot.setRange(xRange=tuple(viewRange[0]), yRange=tuple(viewRange[1]), padding=0)
                    # plot.vb.blockSignals(False)               
    
    # OPTION FUNCTIONS
    def set_ref_p(self):
        ref_p, ok = QInputDialog.getInt(self, 'Set refractory period', 'Set correlogram refractory period (total width in ms)',
                                        value = int(self.ref_p*1000), min=0, max = int(self.window_size*1000), step=1)
        if ok and int(self.ref_p*1000) != ref_p:
            self.ref_p = ref_p/1000
            self.update_ccg(self.cl_list, self.ctl.recording.cl_times) 

    def set_bin_width(self):
        bin_width, ok = QInputDialog.getDouble(self, 'Set bin width', 'Set correlogram bin width (in ms)',
                                        value = int(self.bin_width*1000), min=0.25, max=int(self.window_size*1000))
        if ok and int(self.bin_width*1000) != bin_width:
            self.bin_width = bin_width/1000
            self.acg_cache = {}
            self.update_ccg(self.cl_list, self.ctl.recording.cl_times) 

    def set_window(self):
        window_size, ok = QInputDialog.getInt(self, 'Set window size', 'Set correlogram window size (total width in ms)',
                                        value = int(self.window_size*1000), min=0, max=10000, step=1)
        if ok and int(self.window_size*1000) != window_size:
            self.window_size = window_size/1000
            self.acg_cache = {}
            self.ref_p = min(self.ref_p, self.window_size)
            self.bin_width = min(self.window_size, self.bin_width)
            self.update_ccg(self.cl_list, self.ctl.recording.cl_times) 

    def toggle_labels(self, label):
        self.show_labels = True if label else False
        self.update_ccg(self.cl_list, self.ctl.recording.cl_times) #TODO:replace me with a function that doesn't replot everything

#TODO: change y axis to firing rate
#TODO: only link x axis
class FiringRateView(GraphView):
    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)

        # histogram parameters
        self.n_bins = 100
        self.x_max = self.ctl.recording.data.shape[0]/self.ctl.params['sample_rate']
        self.x_min = 0
        self.bin_width = (self.x_max-self.x_min)/self.n_bins

        self.plots = []

    def update_fr(self, cl_list, cl_times):
        if self.x_max < self.x_min:
            return

        # clear + init for new cl_list
        self.clear()
        self.plots = []
        self.cl_list = cl_list
        n_clust = len(cl_list)

        for i in range(n_clust):
            # set color
            color = (int(colors[i][0]*255), int(colors[i][1]*255), int(colors[i][2]*255))
            pen = mkPen(QColor(color[0], color[1], color[2]), width=0)
            brush = mkBrush(QColor(color[0], color[1], color[2]))

            # calculate histogram
            times_i = cl_times[cl_list[i]]
            fs = self.ctl.params['sample_rate']
            times_i = times_i[(times_i > self.x_min*fs) & (times_i < self.x_max*fs)]
            hist = np.histogram(times_i, self.n_bins)[0]

            # setup plot
            plot = self.addPlot(row=i, col=0)
            self.plots.append(plot)
            plot.vb.sigRangeChanged.connect(self.update_range)

            # plot hist
            x = self.bin_width*np.arange(len(hist)) + self.x_min
            plot.addItem(BarGraphItem(x=x, height=hist, width=self.bin_width, pen=pen, brush=brush))
            plot.showGrid(x=True, y=True, alpha=1)

        self.dock.set_status("[%.2f s, %.2f s]" % (self.x_min, self.x_max))

    def update_range(self, viewBox, viewRange):
        for plot in self.plots:
            if plot and plot.vb is not viewBox:
                # plot.vb.blockSignals(True)  # prevent this update from triggering other updates
                plot.setRange(xRange=tuple(viewRange[0]), yRange=tuple(viewRange[1]), padding=0)
                # plot.vb.blockSignals(False)     

    # OPTION FUNCTIONS
    def set_bin_width(self):
        bin_width, ok = QInputDialog.getDouble(self, 'Set bin width', 'Set histogram bin width (in ms)',
                                        value = int(self.bin_width*1000), min=1)
        if ok and int(self.bin_width*1000) != bin_width:
            self.bin_width = bin_width/1000
            self.n_bins = math.ceil((self.x_max-self.x_min)/self.bin_width)
            self.update_fr(self.cl_list, self.ctl.recording.cl_times)

    def set_n_bins(self):
        n_bins, ok = QInputDialog.getInt(self, 'Set number of bins', 'Set number of histogram bins',
                                        value = self.n_bins, min=1, max=10000)
        if ok and self.n_bins != n_bins:
            self.n_bins = n_bins
            self.bin_width = (self.x_max-self.x_min)/self.n_bins
            self.update_fr(self.cl_list, self.ctl.recording.cl_times)

    def set_x_min(self):
        x_min, ok = QInputDialog.getDouble(self, "Set x min", "Set minimum time value for histogram (in s)",
                                        value = self.x_min, min=0, max=self.ctl.recording.data.shape[0]/self.ctl.params['sample_rate'])
        if ok and self.x_min != x_min:
            self.x_min = x_min
            if self.x_min < self.x_max:
                self.n_bins = math.ceil((self.x_max-self.x_min)/self.bin_width)
            self.update_fr(self.cl_list, self.ctl.recording.cl_times)

    def set_x_max(self):
        x_max, ok = QInputDialog.getDouble(self, "Set x max", "Set maximum time value for histogram (in s)",
                                        value = self.x_max, min=0, max=self.ctl.recording.data.shape[0]/self.ctl.params['sample_rate'])
        if ok and self.x_max != x_max:
            self.x_max = x_max    
            if self.x_min < self.x_max:
                self.n_bins = math.ceil((self.x_max-self.x_min)/self.bin_width)
            self.update_fr(self.cl_list, self.ctl.recording.cl_times)       

class ISIView(GraphView):
    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)

        # histogram parameters
        self.n_bins = 1000
        self.x_max = 1
        self.x_min = 0
        self.bin_width = (self.x_max-self.x_min)/self.n_bins

        self.plots = []

    def update_isi(self, cl_list, cl_times):
        if self.x_max < self.x_min:
            return

        # clear + init for new cl_list
        self.clear()
        self.plots = []
        self.cl_list = cl_list
        n_clust = len(cl_list)

        for i in range(n_clust):
            # set color
            color = (int(colors[i][0]*255), int(colors[i][1]*255), int(colors[i][2]*255))
            pen = mkPen(QColor(color[0], color[1], color[2]), width=0)
            brush = mkBrush(QColor(color[0], color[1], color[2]))

            # calculate histogram
            times_i = cl_times[cl_list[i]]
            isi = np.diff(times_i)
            fs = self.ctl.params['sample_rate']
            isi = isi[(isi > self.x_min*fs) & (isi < self.x_max*fs)]
            hist = np.histogram(isi, self.n_bins)[0]

            # setup plot
            plot = self.addPlot(row=i, col=0)
            self.plots.append(plot)
            plot.vb.sigRangeChanged.connect(self.update_range)

            # plot hist
            x = self.bin_width*np.arange(len(hist)) + self.x_min
            plot.addItem(BarGraphItem(x=x, height=hist, width=self.bin_width, pen=pen, brush=brush))
            plot.showGrid(x=True, y=True, alpha=1)
        
        self.dock.set_status("[%.2f ms, %.2f ms]" % (self.x_min*1000, self.x_max*1000))

    def update_range(self, viewBox, viewRange):
        for plot in self.plots:
            if plot and plot.vb is not viewBox:
                # plot.vb.blockSignals(True)  # prevent this update from triggering other updates
                plot.setRange(xRange=tuple(viewRange[0]), yRange=tuple(viewRange[1]), padding=0)
                # plot.vb.blockSignals(False)     

    # OPTION FUNCTIONS
    def set_bin_width(self):
        bin_width, ok = QInputDialog.getDouble(self, 'Set bin width', 'Set histogram bin width (in ms)',
                                        value = int(self.bin_width*1000), min=1)
        if ok and int(self.bin_width*1000) != bin_width:
            self.bin_width = bin_width/1000
            self.n_bins = math.ceil((self.x_max-self.x_min)/self.bin_width)
            self.update_isi(self.cl_list, self.ctl.recording.cl_times)

    def set_n_bins(self):
        n_bins, ok = QInputDialog.getInt(self, 'Set number of bins', 'Set number of histogram bins',
                                        value = self.n_bins, min=1, max=10000)
        if ok and self.n_bins != n_bins:
            self.n_bins = n_bins
            self.bin_width = (self.x_max-self.x_min)/self.n_bins
            self.update_isi(self.cl_list, self.ctl.recording.cl_times)

    def set_x_min(self):
        x_min, ok = QInputDialog.getDouble(self, "Set x min", "Set minimum time value for histogram (in ms)",
                                        value = self.x_min*1000, min=0, max=10000)
        if ok and self.x_min != x_min/1000:
            self.x_min = x_min/1000
            if self.x_min < self.x_max:
                self.n_bins = math.ceil((self.x_max-self.x_min)/self.bin_width)
            self.update_isi(self.cl_list, self.ctl.recording.cl_times)

    def set_x_max(self):
        x_max, ok = QInputDialog.getDouble(self, "Set x max", "Set maximum time value for histogram (in ms)",
                                        value = self.x_max*1000, min=0, max=10000)
        if ok and self.x_max != x_max/1000:
            self.x_max = x_max/1000    
            if self.x_min < self.x_max:
                self.n_bins = math.ceil((self.x_max-self.x_min)/self.bin_width)
            self.update_isi(self.cl_list, self.ctl.recording.cl_times)       

# TODO: channel map to fix indices
# TODO: labels
# TODO: hover info
class ProbeView(GraphView):
    def __init__(self, controller, **kwargs):
        super().__init__(controller, **kwargs)

        self.mode = 'thresh' # thresh to show channels with amp_frac of peak amplitude, or closest for max_chan closest channels
        self.max_chan = 8
        self.amp_frac = 1/3

    # TODO: smarter offset
    def update_prb(self, cl_list):
        self.clear()
        self.cl_list = cl_list

        base = self.addPlot()
        pen = mkPen(QColor(128,128,128), width=0)
        brush = mkBrush(QColor(128,128,128))
        base.plot(self.ctl.recording.channel_pos[:,0], self.ctl.recording.channel_pos[:,1], pen=None, symbol='o', symbolSize=5, 
            symbolPen=pen, symbolBrush=brush)
        base.hideAxis('left'); base.hideAxis('bottom'); base.hideButtons()

        chans_count = np.zeros(self.ctl.recording.mean_wf.shape[1])

        for i in range(len(cl_list)):
            # find channels with > 1/2 peak amplitude
            mean = self.ctl.recording.mean_wf[cl_list[i]]
            amps = np.max(mean, 1) - np.min(mean, 1)
            chans = np.argsort(amps)[::-1]
            amps_sort = amps[chans]

            peak_amp = amps_sort[0]

            if self.mode == 'thresh':
                chans_cl = chans[amps_sort > peak_amp*self.amp_frac]
                if chans_cl.shape[0] > self.max_chan:
                    chans_cl = chans_cl[:self.max_chan]
            elif self.mode == 'max':
                chans_cl = chans[:self.max_chan]

            # plot
            color = (int(colors[i][0]*255), int(colors[i][1]*255), int(colors[i][2]*255))
            pen = mkPen(QColor(color[0], color[1], color[2]), width=0)
            brush = mkBrush(QColor(color[0], color[1], color[2]))

            base.plot(
                self.ctl.recording.channel_pos[chans_cl,0] + 3*chans_count[chans_cl], # offset for occupied chans
                self.ctl.recording.channel_pos[chans_cl,1], 
                pen=None, symbol='o', symbolSize=7, symbolPen=pen, symbolBrush=brush)

            chans_count[chans_cl] += 1

    def set_amp_frac(self):
        amp_frac, ok = QInputDialog.getDouble(
            self, "Set amp frac", 
            "Set amplitude threshold fraction (only channels with amplitude > amp_frac * peak amplitude will be highlighted)",
            value = self.amp_frac, min=0, max=1)

        if ok and amp_frac != self.amp_frac:
            self.amp_frac = amp_frac   
            self.update_prb(self.cl_list)

    def set_max_chan(self):
        max_chan, ok = QInputDialog.getInt(
            self, "Set max chan", 
            "Set maximum number of channels that can be highlighted",
            value = self.max_chan, min=0, max=self.ctl.recording.mean_wf.shape[1])

        if ok and max_chan != self.max_chan:
            self.max_chan = max_chan   
            self.update_prb(self.cl_list)

    def toggle_mode(self, mode):
        self.mode = 'thresh' if mode else 'max'
        self.update_prb(self.cl_list)
