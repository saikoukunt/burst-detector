from argschema import ArgSchema, ArgSchemaParser 
from argschema.schemas import DefaultSchema
from argschema.fields import InputFile, Nested, InputDir, String, Float, Dict, Int, Bool, NumpyArray, List

# TODO: ADD good_lbls

class AutoMergeParams(ArgSchema):
    # filepaths
    data_filepath = InputFile(required=True, description="Filepath for recording binary")
    KS_folder = InputDir(required=True, description="Kilosort output directory")
    dtype = String(required=False, default="int16", description='Datatype of words in recording binary')
    
    # parameters
    fs = Int(required=False, default=30000, description='Sampling frequency of the recording')
    n_chan = Int(required=False, default=385, description="Number of channels in the recording")
    pre_samples = Int(required=False, default=20, description="Number of samples to extract before the peak of the spike")
    post_samples = Int(required=False, default=62, description="Number of samples to extract after the peak of the spike")
    max_spikes = Int(required=False, default=1000, description="Maximum number of spikes per cluster used to calculate mean waveforms and cross projections (-1 uses all spikes)")
    
    sim_type = String(required=False, default="ae", description="Type of similarity metric to use, must be either \"ae\" (for autoencoder similarity) or \"mean\" (for mean similarity)")
    jitter = Bool(required=False, default=False, description='True if mean similarity calculations should check for time shifts between waveforms')
    jitter_amt = Int(required=False, default=4, description='For time shift checking, number of samples to search in each direction')
    sim_thresh = Float(required=False, default=0.4, description='Similarity threshold for a cluster pair to undergo further stages')
    min_spikes = Int(required=False, default=100, description='Number of spikes threshold for a cluster to undergo further stages.')
    
    # n_iter = Int(required=False, default=50, description="The number of shuffle iterations for the baseline cross-correlation")
    # shuffle_bin_width = Float(required=False, default=0.1, description="The width of bins in seconds for spike train shuffling")
    window_size = Float(required=False, default=0.025, description="The width in seconds of the cross correlogram window.")
    xcorr_bin_width = Float(required=False, default=0.0005, description="The width in seconds of bins for cross correlogram calculation")
    overlap_tol = Float(required=False, default=10/30000, description="Overlap tolerance in seconds. Spikes within the tolerance of the reference spike time will not be counted for cross correlogram calculation")
    max_window = Float(required=False, default=0.25, description="Maximum window size in seconds when searching for enough spikes to construct a cross correlogram")
    min_xcorr_rate = Float(required=False, default=800, description="Spike count threshold (per second) for cross correlograms. Cluster pairs whose cross correlogram spike rate is lower than the threshold will have a penalty applied to their cross correlation metric")
    
    # xcorr_ref_p = Float(required=False, default=0.001, description="Length of refractory period in seconds")
    ref_pers = List(Float, required=False, cli_as_single_argument=True, default=[0.001, 0.002, 0.004], description="List of potential refractory period lengths (in s)")
    max_viol = Float(required=False, default=0.25, description="For refractory period penalty, maximum acceptable proportion (w.r.t uniform acg) of refractory period collisions")

    xcorr_coeff = Float(required=False, default=0.5, description="Coefficient applied to cross correlation metric during final metric calculation")
    ref_pen_coeff = Float(required=False, default=1, description="Coefficient applied to refractory period penalty")
    final_thresh = Float(required=False, default=0.5, description="Final metric threshold for merge decisions")
    max_dist = Int(required=False, default=10, description="Maximum distance between peak channels for a merge to be valid")
    
    ae_pre = Int(required=False, default=10, description="For autoencoder training snippet, number of samples to extract before peak of the spike")
    ae_post = Int(required=False, default=30, description="For autoencoder training snippet, number of samples to extract after peak of the spike")
    ae_chan = Int(required=False, default=8, description="For autoencoder training snippet, number of channels to include")
    ae_noise = Bool(required=False, default=False, description="For autoencoder training, True if autoencoder should explicitly be trained on noise snippets")
    ae_shft = Bool(required=False, default=True, description="For autoencoder training, True if autoencoder should be trained on time-shifted snippets")
    ae_epochs = Int(required=False, default=5, description="Number of epochs to train autoencoder for")
    spikes_path = InputDir(required=False, default=None, description="Path to pre-extracted spikes folder", allow_none=True)
    model_path = InputFile(required=False, default=None, description="Path to pre-trained model", allow_none=True)

class AutomergeGUIParams(ArgSchema):
    # parameters
    pre_samples = Int(required=False, default=20, description="Number of samples to extract before the peak of the spike")
    post_samples = Int(required=False, default=62, description="Number of samples to extract after the peak of the spike")
    max_spikes = Int(required=False, default=1000, description="Maximum number of spikes per cluster used to calculate mean waveforms and cross projections (-1 uses all spikes)")
    
    sim_type = String(required=False, default="ae", description="Type of similarity metric to use, must be either \"ae\" (for autoencoder similarity) or \"mean\" (for mean similarity)")
    jitter = Bool(required=False, default=False, description='True if mean similarity calculations should check for time shifts between waveforms')
    jitter_amt = Int(required=False, default=4, description='For time shift checking, number of samples to search in each direction')
    sim_thresh = Float(required=False, default=0.4, description='Similarity threshold for a cluster pair to undergo further stages')
    min_spikes = Int(required=False, default=100, description='Number of spikes threshold for a cluster to undergo further stages.')
    
    # n_iter = Int(required=False, default=50, description="The number of shuffle iterations for the baseline cross-correlation")
    # shuffle_bin_width = Float(required=False, default=0.1, description="The width of bins in seconds for spike train shuffling")
    window_size = Float(required=False, default=0.025, description="The width in seconds of the cross correlogram window.")
    xcorr_bin_width = Float(required=False, default=0.0005, description="The width in seconds of bins for cross correlogram calculation")
    overlap_tol = Float(required=False, default=10/30000, description="Overlap tolerance in seconds. Spikes within the tolerance of the reference spike time will not be counted for cross correlogram calculation")
    max_window = Float(required=False, default=0.25, description="Maximum window size in seconds when searching for enough spikes to construct a cross correlogram")
    min_xcorr_rate = Float(required=False, default=800, description="Spike count threshold (per second) for cross correlograms. Cluster pairs whose cross correlogram spike rate is lower than the threshold will have a penalty applied to their cross correlation metric")
    
    # xcorr_ref_p = Float(required=False, default=0.001, description="Length of refractory period in seconds")
    ref_pers = List(Float, required=False, cli_as_single_argument=True, default=[0.001, 0.002, 0.004], description="List of potential refractory period lengths (in s)")
    max_viol = Float(required=False, default=0.25, description="For refractory period penalty, maximum acceptable proportion (w.r.t uniform acg) of refractory period collisions")

    xcorr_coeff = Float(required=False, default=0.5, description="Coefficient applied to cross correlation metric during final metric calculation")
    ref_pen_coeff = Float(required=False, default=1, description="Coefficient applied to refractory period penalty")
    final_thresh = Float(required=False, default=0.5, description="Final metric threshold for merge decisions")
    max_dist = Int(required=False, default=10, description="Maximum distance between peak channels for a merge to be valid")
    
    ae_pre = Int(required=False, default=10, description="For autoencoder training snippet, number of samples to extract before peak of the spike")
    ae_post = Int(required=False, default=30, description="For autoencoder training snippet, number of samples to extract after peak of the spike")
    ae_chan = Int(required=False, default=8, description="For autoencoder training snippet, number of channels to include")
    ae_noise = Bool(required=False, default=False, description="For autoencoder training, True if autoencoder should explicitly be trained on noise snippets")
    ae_shft = Bool(required=False, default=False, description="For autoencoder training, True if autoencoder should be trained on time-shifted snippets")
    ae_epochs = Int(required=False, default=5, description="Number of epochs to train autoencoder for")
    spikes_path = InputDir(required=False, default=None, description="Path to pre-extracted spikes folder", allow_none=True)
    model_path = InputFile(required=False, default=None, description="Path to pre-trained model", allow_none=True)
    
class OutputParams(DefaultSchema):
    mean_time = String()
    xcorr_sig_time = String()
    ref_pen_time = String()
    merge_time = String()
    total_time = String()
    num_merges = Int()
    orig_clust = Int()