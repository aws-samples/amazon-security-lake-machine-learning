from typing import Any, Dict, List, Optional
from statsmodels.tsa.seasonal import seasonal_decompose
from kats.detectors.outlier import OutlierDetector
from kats.consts import TimeSeriesData
import pandas as pd
import numpy as np
import logging


class TSATOutlierDetector(OutlierDetector):

    def __init__(self, data: TimeSeriesData, decomp: str = "additive", iqr_mult: float = 3.0) -> None:
        super().__init__(data=data, decomp=decomp, iqr_mult=iqr_mult)

    def __clean_ts__(self, original: pd.DataFrame) -> List:
        """
        Performs detection for a single metric. First decomposes the time series
        and detects outliers when the values in residual time series are beyond the
        specified multiplier times the inter quartile range
        Args:
            original: original time series as DataFrame
        Returns: List of detected outlier timepoints in each metric
        """

        # pyre-fixme[16]: `DataFrame` has no attribute `index`.
        original.index = pd.to_datetime(original.index)

        if pd.infer_freq(original.index) is None:
            # pyre-fixme[9]: original has type `DataFrame`; used as
            #  `Union[pd.core.frame.DataFrame, pd.core.series.Series]`.
            original = original.asfreq("D")
            logging.info("Setting frequency to Daily since it cannot be inferred")

        # pyre-fixme[9]: original has type `DataFrame`; used as `Union[None,
        #  pd.core.frame.DataFrame, pd.core.series.Series]`.
        original = original.interpolate(
            method="polynomial", limit_direction="both", order=3
        )

        # This is a hack since polynomial interpolation is not working here
        if sum((np.isnan(x) for x in original["y"])):
            # pyre-fixme[9]: original has type `DataFrame`; used as `Union[None,
            #  pd.core.frame.DataFrame, pd.core.series.Series]`.
            original = original.interpolate(method="linear", limit_direction="both")

        # Once our own decomposition is ready, we can directly use it here
        result = seasonal_decompose(original, model=self.decomp)
        rem = result.resid
        detrend = original["y"] - result.trend
        strength = float(1 - np.nanvar(rem) / np.nanvar(detrend))
        if strength >= 0.6:
            original["y"] = original["y"] - result.seasonal
        # using IQR as threshold
        resid = original["y"] - result.trend
        resid_q = np.nanpercentile(resid, [25, 75])
        iqr = resid_q[1] - resid_q[0]
        #         limits = resid_q + (self.iqr_mult * iqr * np.array([-1, 1]))

        iqr_mults = resid.abs() / iqr
        outliers_iqr_mults = iqr_mults[iqr_mults > self.iqr_mult]

        #         pdb.set_trace()

        #         outliers = resid[(resid >= limits[1]) | (resid <= limits[0])]
        self.outliers_index = outliers_index = list(outliers_iqr_mults.index)
        return list(zip(outliers_index, outliers_iqr_mults))