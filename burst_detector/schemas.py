from argschema import ArgSchema, ArgSchemaParser 
from argschema.schemas import DefaultSchema
from argschema.fields import InputFile, Nested, InputDir, String, Float, Dict, Int, Bool, NumpyArray, List

class AutoMergeParams(ArgSchema):
    # filepaths
    data_filepath = InputFile(required=True, description="Filepath for recording binary")
    KS_folder = InputDir(required=True, description="Kilosort output directory")
    
    # parameters
    calc_means = Bool(required=False, default=False, description='True if mean waveforms should be calculated, False if they are being passed in in mean_waveforms.py')
    fs = Int(required=False, default=30000, description='sampling frequency of the recording')
    n_chan = Int(required=False, default=385, description="number of channels in the recording")
    pre_samples = Int(required=False, default=20, description="number of samples to extract before the peak of the spike")
    post_samples = Int(required=False, default=62, description="number of samples to extract after the peak of the spike")
    max_spikes = Int(required=False, default=1000, description="maximum number of spikes per cluster used to calculate mean waveforms and cross projections (-1 uses all spikes)")
    
    skip_cross_sim = Bool(required=False, default=False, description="True if cross sim should be skipped")
    
    jitter = Bool(required=False, default=False, description='True if similarity calculations should check for time shifts between waveforms')
    jitter_amt = Int(required=False, default=4, description='For time shift checking, number of samples to search in each direction')
    sim_thresh = Float(required=False, default=0.6, description='Mean waveformm inner product threshold for a cluster pair to undergo further stages')
    sp_num_thresh = Int(required=False, default=100, description='Number of spikes threshold for a cluster pair to undergo further stages. At least one cluster in the pair must contain more than sp_num_thresh spikes')
    
    n_iter = Int(required=False, default=50, description="The number of shuffle iterations for the baseline cross-correlation")
    shuffle_bin_width = Float(required=False, default=0.1, description="The width of bins in seconds for spike train shuffling")
    window_size = Float(required=False, default=0.025, description="The width in seconds of the cross correlogram window.")
    xcorr_bin_width = Float(required=False, default=0.001, description="The width in seconds of bins for cross correlogram calculation")
    overlap_tol = Float(required=False, default=10/30000, description="Overlap tolerance in seconds. Spikes within the tolerance of the reference spike time will not be counted for cross correlogram calculation")
    max_window = Float(required=False, default=0.25, description="Maximum window size in seconds when searching for enough spikes to construct a cross correlogram")
    min_xcorr_rate = Float(required=False, default=.5/.001, description="Spike rate threshold in Hz for cross correlograms. Cluster pairs whose cross correlogram spike rate is lower than the threshold willl have a penalty applied to their cross correlation metric")
    
    xcorr_ref_p = Float(required=False, default=0.001, description="Length of refractory period in seconds")
    ref_pers = List(Float, required=False, cli_as_single_argument=True, default=[0.002, 0.005, 0.010], description="List of potential refractory period lengths (in s)")
    max_viol = Float(required=False, default=0.01, description="For refractory period penalty, maximum acceptable proportion of refractory period collisions (new violations due to merge)")

    xcorr_coeff = Float(required=False, default=0.5, description="Coefficient applied to cross correlation metric during final metric calculation")
    ref_pen_coeff = Float(required=False, default=1, description="Coefficient applied to refractory period penalty")
    final_thresh = Float(required=False, default=0.7, description="Final metric threshold for merge decisions")
    max_dist = Int(required=False, default=10, description="Maximum distance between peak channels for a merge to be valid")
    
class OutputParams(DefaultSchema):
    mean_time = String()
    cache_spikes_time = String()
    cross_sim_time = String()
    xcorr_sig_time = String()
    ref_pen_time = String()
    merge_time = String()
    total_time = String()
    num_merges = Int()
    orig_clust = Int()