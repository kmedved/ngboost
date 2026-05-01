import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.utils.concordance import (
    _concordance_ratio,
    _concordance_summary_statistics,
)
from matplotlib import pyplot as plt


def calibration_regression(Forecast, Y, bins=11, eps=1e-3):
    """
    Calculate calibration in the regression setting.
    """
    pctles = np.linspace(eps, 1 - eps, bins)
    observed = np.zeros_like(pctles)
    for i, pctle in enumerate(pctles):
        icdfs = Forecast.ppf(pctle).reshape(Y.shape)
        observed[i] = np.mean(Y < icdfs)
    slope, intercept = np.polyfit(pctles, observed, deg=1)
    return pctles, observed, slope, intercept


def calibration_time_to_event(Forecast, T, E):
    """
    Calculate calibration in the time-to-event setting, with integral transform and KM.
    """
    cdfs = Forecast.cdf(T)
    kmf = KaplanMeierFitter()
    kmf.fit(cdfs, E)
    idxs = np.round(np.linspace(0, len(kmf.survival_function_) - 1, 11))
    preds = np.array(kmf.survival_function_.iloc[idxs].index)
    obs = 1 - np.array(kmf.survival_function_.iloc[idxs].KM_estimate)
    slope, intercept = np.polyfit(preds, obs, deg=1)
    return preds, obs, slope, intercept


def calculate_calib_error(predicted, observed):
    return np.sum((predicted - observed) ** 2) / len(predicted)


def plot_pit_histogram(predicted, observed, **kwargs):
    plt.bar(
        x=predicted[1:],
        height=np.diff(observed),
        width=-np.diff(predicted),
        align="edge",
        fill=False,
        edgecolor="black",
        **kwargs,
    )
    plt.xlim((0, 1))
    plt.xlabel("Probability Integral Transform")
    plt.ylabel("Density")
    plt.axhline(1.0 / (len(predicted) - 1), linestyle="--", color="grey")
    plt.title("PIT Histogram")


def plot_calibration_curve(predicted, observed):
    """
    Plot calibration curve.
    """
    slope, intercept = np.polyfit(predicted, observed, deg=1)
    plt.plot(predicted, observed, "o", color="black")
    plt.plot(
        np.linspace(0, 1),
        np.linspace(0, 1) * slope + intercept,
        "--",
        label=f"Slope: {slope:.2f}, Intercept: {intercept:.2f}",
        alpha=0.5,
        color="black",
    )
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), "--", color="grey", alpha=0.5)
    plt.xlabel("Predicted CDF")
    plt.ylabel("Observed CDF")
    plt.title("Calibration Plot")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend(loc="upper left")


def calculate_concordance_dead_only(preds, Y, E):
    """
    Calculate C-statistic for only cases where outcome is uncensored.
    """
    return calculate_concordance_naive(
        np.array(preds[E == 1]), np.array(Y[E == 1]), np.array(E[E == 1])
    )


def calculate_concordance_naive(preds, Y, E):
    """
    Calculate Harrell's C-statistic in the presence of censoring.

    Cases:
    - (c=0, c=0): both uncensored, can compare
    - (c=0, c=1): can compare if true censored time > true uncensored time
    - (c=1, c=0): can compare if true censored time > true uncensored time
    - (c=1, c=1): both censored, cannot compare
    """
    event_mask = np.asarray(E, dtype=bool)
    times = np.asarray(Y)
    predictions = np.asarray(preds)
    concordant, tied_pred, admissible = _concordance_summary_statistics(
        times, predictions, event_mask
    )

    event_times = times[event_mask]
    event_preds = predictions[event_mask]
    censored_times = times[~event_mask]
    censored_preds = predictions[~event_mask]

    _, event_counts = np.unique(event_times, return_counts=True)
    admissible += np.sum(event_counts * (event_counts - 1) // 2)

    if len(event_times) > 0:
        event_pairs = np.rec.fromarrays([event_times, event_preds])
        _, event_pair_counts = np.unique(event_pairs, return_counts=True)
        tied_pred += np.sum(event_pair_counts * (event_pair_counts - 1) // 2)

    event_order = np.lexsort((event_preds, event_times))
    censored_order = np.lexsort((censored_preds, censored_times))
    sorted_event_times = event_times[event_order]
    sorted_event_preds = event_preds[event_order]
    sorted_censored_times = censored_times[censored_order]
    sorted_censored_preds = censored_preds[censored_order]

    event_start = censored_start = 0
    while event_start < len(sorted_event_times) and censored_start < len(
        sorted_censored_times
    ):
        event_time = sorted_event_times[event_start]
        censored_time = sorted_censored_times[censored_start]
        if event_time < censored_time:
            event_start = np.searchsorted(sorted_event_times, event_time, side="right")
            continue
        if censored_time < event_time:
            censored_start = np.searchsorted(
                sorted_censored_times, censored_time, side="right"
            )
            continue

        event_stop = np.searchsorted(sorted_event_times, event_time, side="right")
        censored_stop = np.searchsorted(
            sorted_censored_times, censored_time, side="right"
        )
        same_time_event_preds = sorted_event_preds[event_start:event_stop]
        same_time_censored_preds = sorted_censored_preds[censored_start:censored_stop]
        admissible -= len(same_time_event_preds) * len(same_time_censored_preds)
        concordant -= np.searchsorted(
            same_time_event_preds, same_time_censored_preds, side="left"
        ).sum()
        tied_pred -= (
            np.searchsorted(
                same_time_event_preds, same_time_censored_preds, side="right"
            )
            - np.searchsorted(
                same_time_event_preds, same_time_censored_preds, side="left"
            )
        ).sum()
        event_start = event_stop
        censored_start = censored_stop

    return _concordance_ratio(concordant, tied_pred, admissible)
