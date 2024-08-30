import cProfile
from typing import Any

import matplotlib.pyplot as plt
from argschema import ArgSchemaParser

import burst_detector as bd


def main() -> None:
    from burst_detector import AutoMergeParams, OutputParams

    mod = ArgSchemaParser(schema_type=AutoMergeParams)
    # TODO fix schema
    if mod.args.get("max_spikes") is None:
        mod.args["max_spikes"] = 1000
    mst: str
    xct: str
    rpt: str
    mt: str
    tt: str
    num_merge: int
    oc: int
    mst, xct, rpt, mt, tt, num_merge, oc = bd.run_merge(mod.args)

    output: dict[str, Any] = {
        "mean_time": mst,
        "xcorr_time": xct,
        "ref_pen_time": rpt,
        "merge_time": mt,
        "total_time": tt,
        "num_merges": num_merge,
        "orig_clust": oc,
    }

    mod.output(output)


if __name__ == "__main__":
    main()
