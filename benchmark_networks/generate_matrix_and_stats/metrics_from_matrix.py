import numpy as np
import utils.mask_utils
import utils.utils_OF as utils_OF
##FULL FRAME METRICS MATRIX
def full_frame_metrics(out,Tout_180,target,test_on_symmetric_data=False):
    # MAGNITUDE
    # L1 magnitude


    target_mag_l1 = abs(target.copy())
    # target magnitude
    target_magL2_sq = np.square(target.copy())  # elementwise square
    target_magL2_sq_summed = target_magL2_sq[:, :, 0] + target_magL2_sq[:, :, 1]
    target_magL2 = np.sqrt(target_magL2_sq_summed)

    EPE_L1 = abs(out.copy( ) -target.copy())
    EPE = utils_OF.EPEmatrix(out ,target)
    EPE_sq = utils_OF.EPE_squared_matrix(out ,target)
    EPE_180 = utils_OF.EPEmatrix(Tout_180, -target)
    ##
    I_180 = out.copy() + Tout_180.copy()
    Iu = I_180[: ,: ,0]
    Iv = I_180[: ,: ,1]



    ##writing means to row
    row = {}
    row['EPE'] = np.nanmean(EPE)
    row['EPE_180'] = np.nanmean(EPE_180)
    row['EPE_u'] = np.nanmean(abs(EPE_L1[: ,: ,0]))
    row['EPE_v'] = np.nanmean(abs(EPE_L1[: ,: ,1]) )

    row['cos_sim'] = utils_OF.cos_sim(out,target)[0]
    row['spatium'] = utils_OF.spatium_error(out,target)[0]

    row['G_mag_u'] = np.nanmean(target_mag_l1 ,axis=(0 ,1))[0]
    row['G_mag_v'] = np.nanmean(target_mag_l1 ,axis=(0 ,1))[1]
    row['G_mag_L2'] = np.nanmean(target_magL2)#stefano.mean())

    row['Iu_m1'] = np.nanmean(abs(Iu.copy()))
    row['Iv_m1'] = np.nanmean(abs(Iv.copy()))

    # Euclidean distance

    if test_on_symmetric_data==True:

        q2q3, q1q4 = np.hsplit(out.copy(), 2)
        q2, q3 = np.vsplit(q2q3.copy(), 2)
        q1, q4 = np.vsplit(q1q4.copy(), 2)

        q1_hf=np.fliplr(q1.copy())
        #q1_hf[...,0]=q1_hf[...,0]*(-1)
        q3vf=np.flipud(q3.copy())
        #q1_hf[..., 1] = q1_hf[..., 1] * (-1)
        q4_180=np.rot90(q4.copy(),2,axes=(1,0))
        Im = q2 +  q4_180 +q1_hf+q3vf

        #row['I_L2_m1'] = np.sqrt(np.sum(np.square(np.sum(Im.copy(),axis=(0,1)))))
        # #row['I_L2_m1'] = abs(np.mean(out))
        row['Iu_m1'] = np.mean(abs(Im[...,0]))
        row['Iv_m1'] = np.mean(abs(Im[...,1]))

        I_180_squared = np.square(Im.copy())
        I_180_squared_summed = I_180_squared[:, :, 0] + I_180_squared[:, :, 1]
        row['I_L2_m1'] = np.mean(np.sqrt(I_180_squared_summed.copy()))

        # print(row['G_mag_L2'])
        # print(row['EPE'])
        # print(row['I_L2_m1'], np.mean(np.sqrt(I_180_squared_summed.copy())))
        # print('K')
    else:

        I_180_squared = np.square(I_180.copy())
        I_180_squared_summed = I_180_squared[:, :, 0] + I_180_squared[:, :, 1]
        row['I_L2_m1'] = np.nanmean(np.sqrt(I_180_squared_summed.copy()))

    return row


def masked_metrics(out, out_star, target, thresholds, include_lower_bound=True, \
                       include_upper_bound=False):

    out[out==np.nan]=0
    out_star[out_star == np.nan] = 0
    target[target== np.nan]=0


    row = {}
    # target magnitude
    target_magL2_sq = np.square(target.copy())  # elementwise square
    target_magL2_sq_summed = target_magL2_sq[:, :, 0] + target_magL2_sq[:, :, 1]
    target_magL2 = np.sqrt(target_magL2_sq_summed)

    EPE = utils_OF.EPEmatrix(out ,target)
    EPE_L1 = abs(out.copy() - target.copy())

    ##
    I_180 = out + out_star
    Iu = I_180[: ,: ,0]
    Iv = I_180[: ,: ,1]
    ## masked magnitudes
    I_180_squared = np.square(I_180.copy())
    I_180_squared_summed = I_180_squared[: ,: ,0] + I_180_squared[: ,: ,1]

    # matrix
    # target_mag_l1,target_magL2,EPE_L1,EPE,Iu,Iv,Iu_180,Iv_180
    EPE_sum_msk, G_mag_L2_masked_sum , G_mag_L2_pixel_counts = utils.mask_utils.mask_matrix_over_target(EPE, target_magL2, thresholds=thresholds
                                                                                                      ,include_lower_bound = include_lower_bound
                                                                                                      ,include_upper_bound=include_upper_bound)
    EPE_u_sum_msk, G_mag_L2_masked_sum , G_mag_L2_pixel_counts = utils.mask_utils.mask_matrix_over_target(EPE_L1[:,:,0], target_magL2, thresholds=thresholds
                                                                                                      ,include_lower_bound = include_lower_bound
                                                                                                      ,include_upper_bound=include_upper_bound)
    EPE_v_sum_msk, G_mag_L2_masked_sum , G_mag_L2_pixel_counts = utils.mask_utils.mask_matrix_over_target(EPE_L1[:,:,1], target_magL2, thresholds=thresholds
                                                                                                      ,include_lower_bound = include_lower_bound
                                                                                                      ,include_upper_bound=include_upper_bound)


    Iu_sum_m1_msk, _, _ = utils.mask_utils.mask_matrix_over_target(abs(Iu), target_magL2 ,thresholds=thresholds
                                                                      ,include_lower_bound = include_lower_bound
                                                                      ,include_upper_bound=include_upper_bound)
    Iv_sum_m1_msk, _, _ = utils.mask_utils.mask_matrix_over_target(abs(Iv), target_magL2, thresholds=thresholds
                                                                      ,include_lower_bound = include_lower_bound
                                                                      ,include_upper_bound=include_upper_bound)


    I_L2_sum_m1_msk, _, _ = utils.mask_utils.mask_matrix_over_target(np.sqrt(I_180_squared_summed), target_magL2,
                                                                   thresholds=thresholds,
                                                                   include_lower_bound=include_lower_bound,
                                                                   include_upper_bound=include_upper_bound)

##continue from here
    # labels_L2 = ['EPE','G_mag_L2','PIXEL_COUNT','Iu_Tlr_m2','Iv_Tud_m2','Iu_T180_m2','Iv_T180_m2','Iuv_T180_L2']
    # labels_L1 = ['EPE_L1','G_mag_L1_tuple', 'G_mag_L2', 'PIXEL_COUNT', 'Iu_Tlr_m2', 'Iv_Tud_m2', 'Iu_T180_m2', 'Iv_T180_m2', 'Iuv_T180_L2']
    ##l2 stats
    row['thresholds'] = thresholds
    row['EPE_sum_msk'] = EPE_sum_msk
    row['EPEu_sum_msk'] = EPE_u_sum_msk
    row['EPEv_sum_msk'] = EPE_v_sum_msk
    row['G_mag_L2_masked_sum'] = G_mag_L2_masked_sum
    row['G_mag_L2_PIXEL_COUNT'] = G_mag_L2_pixel_counts
    row['I_L2_sum_m1_msk'] = I_L2_sum_m1_msk
    row['Iv_sum_m1_msk'] = Iv_sum_m1_msk
    row['Iu_sum_m1_msk'] = Iu_sum_m1_msk


    return row





def Tud_Tlr_T180_metrics(out, Tout_lr, Tout_ud, Tout_180):
    row = {}

    Iu = out[:,:,0] + Tout_lr[:,:,0]
    Iv = out[:,:,1] + Tout_ud[:,:,1]

    I_180 = out + Tout_180
    Iu_180 = I_180[:,:,0]
    Iv_180 = I_180[:,:,1]


    row['Iu_Tlr_m1'] = np.mean(abs(Iu.copy()))
    row['Iv_Tud_m1'] = np.mean(abs(Iv.copy()))

    row['Iu_T180_m1'] = np.mean(abs(Iu_180.copy()))
    row['Iv_T180_m1'] = np.mean(abs(Iv_180.copy()))


    row['Iu_Tlr_m2_mean'] = np.sqrt(np.divide(np.sum(np.square(Iu.copy())),Iu.size))
    row['Iv_Tud_m2_mean'] = np.sqrt(np.divide(np.sum(np.square(Iv.copy())),Iu.size))

    row['Iu_T180_m2'] = np.sqrt(np.divide(np.sum(np.square(Iu_180.copy())),Iu.size))
    row['Iv_T180_m2'] = np.sqrt(np.divide(np.sum(np.square(Iv_180.copy())),Iu.size))
    #row['pixel_number'] = Iu.size


    return row

def split_matrix_in_quadrants(mat):
    q1 = mat[:mat.shape[0] // 2, mat.shape[1] // 2:, :]
    q2 = mat[:mat.shape[0] // 2, :mat.shape[1] // 2, :]
    q3 = mat[mat.shape[0] // 2:, :mat.shape[1] // 2, :]
    q4 = mat[mat.shape[0] // 2:, mat.shape[1] // 2:, :]
    assert q1.shape == q2.shape == q3.shape == q4.shape
    return q1, q2, q3, q4

def quarter_metrics(out, out_star, target):
    ### mask matrix on quadrants
    Tq1,Tq2,Tq3,Tq4 = split_matrix_in_quadrants(target)
    Oq1, Oq2, Oq3, Oq4 = split_matrix_in_quadrants(out)
    O_star_q1, O_star_q2, O_star_q3, O_star_q4 = split_matrix_in_quadrants(out_star)
    all_row = []
    row = {}
    for T, O, O_star, label in zip([Tq1, Tq2, Tq3, Tq4], [Oq1, Oq2, Oq3, Oq4],[O_star_q1, O_star_q2, O_star_q3, O_star_q4],\
                                                             ['q1', 'q2', 'q3', 'q4']):


        target_mag_l1 = abs(T.copy())
        # target magnitude
        target_magL2_sq = np.square(T.copy())  # elementwise square
        target_magL2_sq_summed = target_magL2_sq[:, :, 0] + target_magL2_sq[:, :, 1]
        target_magL2 = np.sqrt(target_magL2_sq_summed)

        EPE_L1 = abs(O.copy() - T.copy())
        EPE = utils_OF.EPEmatrix(O,T)
        EPE_sq = utils_OF.EPE_squared_matrix(O, T)

        ##
        I_180 = O + O_star
        Iu = I_180[:, :, 0]
        Iv = I_180[:, :, 1]

        ##writing means to row

        row['EPE'+ '_' +label] = np.mean(EPE)
        row['EPE_u'+ '_' +label] = np.mean(abs(EPE_L1[:, :, 0]))
        row['EPE_v'+ '_' +label] = np.mean(abs(EPE_L1[:, :, 1]))

        row['cos_sim'+ '_' +label] = utils_OF.cos_sim(out, target)[0]
        row['spatium'+ '_' +label] = utils_OF.spatium_error(out, target)[0]

        row['G_mag_u'+ '_' +label] = np.mean(target_mag_l1, axis=(0, 1))[0]
        row['G_mag_v'+ '_' +label] = np.mean(target_mag_l1, axis=(0, 1))[1]
        row['G_mag_L2'+ '_' +label] = np.mean(target_magL2.mean())

        row['Iu_m1'+ '_' +label] = np.mean(abs(Iu.copy()))
        row['Iv_m1'+ '_' +label] = np.mean(abs(Iv.copy()))

        # euclidian distance
        I_180_squared = np.square(I_180.copy())
        I_180_squared_summed = I_180_squared[:, :, 0] + I_180_squared[:, :, 1]
        row['I_L2_m1'] = np.mean(np.sqrt(I_180_squared_summed.copy()))

        row1 ={}
        #all_row.append(row)
        row1['EPE'] = np.mean(EPE)
        row1['EPE_u'] = np.mean(abs(EPE_L1[:, :, 0]))
        row1['EPE_v'] = np.mean(abs(EPE_L1[:, :, 1]))

        row1['cos_sim'] = utils_OF.cos_sim(out, target)[0]
        row1['spatium'] = utils_OF.spatium_error(out, target)[0]

        row1['G_mag_u'] = np.mean(target_mag_l1, axis=(0, 1))[0]
        row1['G_mag_v'] = np.mean(target_mag_l1, axis=(0, 1))[1]
        row1['G_mag_L2'] = np.mean(target_magL2.mean())

        row1['Iu_m1'] = np.mean(abs(Iu.copy()))
        row1['Iv_m1'] = np.mean(abs(Iv.copy()))

        # euclidian distance
        I_180_squared = np.square(I_180.copy())
        I_180_squared_summed = I_180_squared[:, :, 0] + I_180_squared[:, :, 1]
        row1['I_L2_m1'] = np.mean(np.sqrt(I_180_squared_summed.copy()))
        row1['quadrant'] = label
        all_row.append(row1)

    return row,all_row