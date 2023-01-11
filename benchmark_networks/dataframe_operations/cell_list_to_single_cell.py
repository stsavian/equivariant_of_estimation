import numpy as np
import pandas as pd

def masked_stats_to_dataframe(data,msklist,list_counts = 'G_mag_L2_PIXEL_COUNT',mean_type='m1'):
    masked_means = {}

    for metric in msklist:
        df = data[metric].copy()
        df = df.apply(lambda x: pd.Series(x))
        #df.columns = thresholds_list
        #masked_means[metric] = np.round(df.mean().tolist(),2)
        if mean_type =='m1':

            if metric == 'thresholds':#
                print(metric)
                masked_means[metric] = data[metric].apply(lambda x: pd.Series(x)).mean()
                #masked_means[metric] =#continue here
            else:
                masked_means[metric] = np.round(np.divide(df.sum(), data[list_counts].apply(lambda x: pd.Series(x)).sum()).tolist(), 2)#new
        elif mean_type =='m2':
            masked_means[metric] = np.round(\
                np.sqrt(np.divide(df.sum(), data[list_counts].apply(lambda x: pd.Series(x)).sum()).tolist()), 2)
        masked_means['thresholds'] = data['thresholds'].iloc[0][:-1]
        masked_means['px_count'] = data[list_counts].apply(lambda x: pd.Series(x)).sum()
        out = pd.DataFrame(masked_means).T
        out.columns = data['thresholds'].iloc[0][:-1]
    return out

