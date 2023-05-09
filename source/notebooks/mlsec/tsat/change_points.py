from ..tsat import *

def plot_multiple_change_points(idxs, ts_list, change_points_detected):
    for ts_i in idxs:
        ts = ts_list[ts_i]
        cps = change_points_detected[ts_i]
        ax = plot_single_timeseries(ts, ylabel=f'idx = {ts_i}')
        ax = plot_vertical_timestamp_lines(ax, [cp[0].start_time for cp in cps])
        plt.show()