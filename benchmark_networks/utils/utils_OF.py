import numpy as np
#import torch
import torch
def EPE(flo,gnd):
    #flo = readFlow(flo_pth)
    #gnd = readFlow(gnd_pth)
    flo_u = flo[:,:,0]
    flo_v = flo[:,:,1]
    gnd_u = gnd[:,:,0]
    gnd_v = gnd[:,:,1]
    epe = np.sqrt(np.square(flo_u - gnd_u) + np.square(flo_v - gnd_v))
    return np.mean(epe)

def EPEmatrix(flo,gnd):
    #flo = readFlow(flo_pth)
    #gnd = readFlow(gnd_pth)
    flo_u = flo[:,:,0]
    flo_v = flo[:,:,1]
    gnd_u = gnd[:,:,0]
    gnd_v = gnd[:,:,1]
    epe = np.sqrt(np.square(flo_u - gnd_u) + np.square(flo_v - gnd_v))
    return epe

def EPE_squared_matrix(flo,gnd):
    #flo = readFlow(flo_pth)
    #gnd = readFlow(gnd_pth)
    flo_u = flo[:,:,0]
    flo_v = flo[:,:,1]
    gnd_u = gnd[:,:,0]
    gnd_v = gnd[:,:,1]
    epe = np.square(flo_u - gnd_u) + np.square(flo_v - gnd_v)
    return epe





def normalize_flow(flow):
    temp = flow.copy()
    aux = np.square(temp)
    aux_norms = np.sqrt(aux[...,0]+aux[...,1])
    norm = np.stack((aux_norms, aux_norms), axis=-1)
    #beware of zeroes
    eps= 1e-8
    norm[norm==0] =eps
    temp[norm==0] +=eps
    try:
        flow_norm = np.divide(temp,norm)
    except Exception as e:
        print(e)

    return flow_norm

def mean_mag_matrix(target):
    #target magnitude
    target_magL2 = target.copy()
    target_magL2 = np.square(target_magL2)  # elementwise square
    target_magL2 = target_magL2[:, :, 0] + target_magL2[:, :, 1]
    target_magL2 = np.sqrt(target_magL2)
    target_magL2 = np.mean(target_magL2, axis=(0, 1))
    return target_magL2


def cos_sim(out_flo,target):

    sim_m = torch.nn.functional.cosine_similarity(torch.from_numpy(out_flo.copy()), torch.from_numpy(target.copy()), dim=2, eps=1e-8)
    epsilon = 1e-8
    sim_clamped_m = torch.acos(torch.clamp(sim_m, -1 + epsilon, 1 - epsilon))
    sim_m_clamped_deg = np.rad2deg(sim_clamped_m)
    #sim_deg = np.rad2deg(sim_clamped_m)
    sim_deg = np.nanmean(abs(sim_m_clamped_deg))

    return abs(sim_deg),abs(sim_m_clamped_deg)

def spatium_error(out_flo,target):

    sim_m = torch.nn.functional.cosine_similarity(torch.from_numpy(out_flo.copy()), torch.from_numpy(target.copy()), dim=2, eps=1e-8)
    epsilon = 1e-8
    sim_clamped_m = torch.clamp(sim_m, -1 + epsilon, 1 - epsilon)
    theta_m = torch.acos(sim_clamped_m)
    gnd_mag = torch.norm(torch.from_numpy(target.copy()), 2, 2)
    s_m = torch.mul(gnd_mag,theta_m).numpy()
    s = np.nanmean(s_m)
    #sim_m_clamped_deg = np.rad2deg(sim_clamped_m)
    #sim_deg = np.rad2deg(sim_clamped_m)
    #sim_deg = torch.mean(abs(sim_m_clamped_deg))

    return abs(s),abs(s_m)