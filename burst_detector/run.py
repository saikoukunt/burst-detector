import burst_detector as bd
from argschema import ArgSchemaParser
import matplotlib.pyplot as plt
import cProfile

def main():
    from burst_detector import AutoMergeParams, OutputParams
    
    mod = ArgSchemaParser(schema_type=AutoMergeParams)
    mst, xct, rpt, mt, tt, num_merge, oc = bd.run_merge(mod.args)
    # cProfile.runctx("bd.run_merge(params)",{"bd":bd},{'params':mod.args})
    
    output = {
        'mean_time': mst,
        'xcorr_time': xct, 
        'ref_pen_time': rpt,
        'merge_time': mt,
        'total_time': tt,
        'num_merges': num_merge,
        'orig_clust': oc
    }
    
    mod.output(output)
    
    
if __name__ == "__main__":
    main()