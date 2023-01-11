import numpy as np
def mask_matrix_over_target(Imag, target_mag, thresholds=[0, 2, 5, 10, 20, 30, 50, 1000],include_lower_bound = False,include_upper_bound=True):
    """Return entries for dataframe
    Inputs: Imag one layer matrix. target_mag, one layer matrix


    """
    #mask groundtruth matrix
    #returns dict of masks
    tr_dict = {}
    tr_pixel_count = {} #new
    for i in range(1, len(thresholds)):
        if include_upper_bound:
            mask = (abs(target_mag) <= thresholds[i])
        else:
            mask = (abs(target_mag) < thresholds[i])
        if include_lower_bound:
            mask_1 = (abs(target_mag) >= thresholds[i - 1])
        else:
            mask_1 = (abs(target_mag) > thresholds[i - 1])
        mask_tot = (mask & mask_1)
        tr_dict[thresholds[i]] = mask_tot
        tr_dict.update({thresholds[i]: mask_tot})
        tr_pixel_count.update({thresholds[i]:np.sum(mask_tot == True)}) #new

    #mask I and groundtruth
    I_mag_dict = {}
    target_mag_dict = {}
    for i in range(1, len(thresholds)):
        I_thr = Imag[tr_dict[thresholds[i]]] # check
        I_mag_dict.update({thresholds[i]: I_thr})

        gnd_mag_thr = target_mag[tr_dict[thresholds[i]]]  # check
        target_mag_dict.update({thresholds[i]: gnd_mag_thr})

    #save masked pixels
    I_mag_means = []
    for i in range(1, len(thresholds)):
        if I_mag_dict[thresholds[i]].size != 0:
            I_mag_means.append(np.sum(I_mag_dict[thresholds[i]])) #new all means changed with sum
        elif I_mag_dict[thresholds[i]].size == 0:
            I_mag_means.append(np.nan)

    target_mag_sums = []
    target_pixel_counts = []#new
    for i in range(1, len(thresholds)):
        if target_mag_dict[thresholds[i]].size != 0:
            target_mag_sums.append(np.sum(target_mag_dict[thresholds[i]]))#new
            target_pixel_counts.append(tr_pixel_count[thresholds[i]])#new
        elif target_mag_dict[thresholds[i]].size == 0:
            target_mag_sums.append(np.nan)
            target_pixel_counts.append(np.nan)

    title = []
    I_sums = []

    for i in range(1, len(thresholds)):
        title.append(thresholds[i])
        if I_mag_dict[thresholds[i]].size != 0:
            I_sums.append(I_mag_dict[thresholds[i]].sum())#new
        else:
            I_sums.append(np.nan)



    return  I_sums, target_mag_sums,target_pixel_counts#new