from typing import Any
import burst_detector as bd
from argschema import ArgSchemaParser
import matplotlib.pyplot as plt
import cProfile

def main() -> None:
    from burst_detector import AutoMergeParams, OutputParams
    
    mod = ArgSchemaParser(schema_type=AutoMergeParams)
    mst: str; xct: str; rpt: str; mt: str; tt: str; num_merge: int; oc: int
    mst, xct, rpt, mt, tt, num_merge, oc = bd.run_merge(mod.args)
    
    output: dict[str, Any] = {
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