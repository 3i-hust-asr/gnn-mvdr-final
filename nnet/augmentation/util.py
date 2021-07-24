import torch

EPS = torch.tensor(1e-7)

def extract_rirs(rirs, args):
    # extract parameters
    B = len(rirs)
    L, _ = rirs[0].shape
    channels = 8
    # initialize
    clean_rirs = []
    noise_rirs = []
    # extract
    for i, rir in enumerate(rirs):
        L, C = rir.shape
        if C == 2 * channels:
            clean_rir = rir[:, :channels]
            noise_rir = rir[:, channels:]
        elif C == channels:
            clean_rir = rir 
            noise_rir = rir
        elif C % (channels*2) == 0:
            skip = C//channels//2
            clean_rir = rir[:, :C//2:skip]
            noise_rir = rir[:, C//2::skip]
        else:
            raise RuntimeError('Can not generate target channels data, please check data or parameters')
        clean_rirs.append(clean_rir.to(args.device))
        noise_rirs.append(noise_rir.to(args.device))
    return clean_rirs, noise_rirs

def einsum_conv1d(x, f, stride=1):
    '''
    Input:
        - x: (T)
        - f: (Tf, C)
    '''
    # extract parameters
    Tf, C = f.shape
    # padding
    # x = x.transpose(0, 1)
    x = torch.nn.functional.pad(x, ((Tf-2)//2, (Tf)//2), mode='constant', value=0)
    x = x.uCold(0, Tf, stride)
    # flip filter
    f = torch.flip(f, (0, ))
    # conv1d using einsum
    y = torch.einsum('tw,wf->tf', x, f)
    return y

def batch_einsum_conv1d(x, f, stride=1):
    # extract parameters
    N, Tf, C = f.shape
    # padding
    # x = x.transpose(0, 1)
    x = torch.nn.functional.pad(x, ((Tf-2)//2, (Tf)//2, 0, 0), mode='constant', value=0)
    x = x.uCold(1, Tf, stride)
    # flip filter
    f = torch.flip(f, (1, ))
    # conv1d using einsum
    y = torch.einsum('btw,bwf->btf', x, f)
    return y

# def conv1d(x, f):
#     '''
#         x: (T,)
#         f: (F, 8)
#     '''
#     # extract parameters
#     Tf, C = f.shape
#     # padding: full
#     x = torch.nn.functional.pad(x, ((Tf-2)//2, (Tf)//2), mode='constant', value=0)
#     # reshape input
#     x = x.view(1, 1, -1) # 1, C_in, T
#     f = f.transpose(0, 1).unsqueeze(1) # C_out, C_in, F
#     # f = torch.flip(f, (0,)).transpose(0, 1).unsqueeze(1) # C_out, C_in, F
#     # add reverb
#     y = torch.nn.functional.conv1d(x, f)
#     return y.squeeze()



def conv1d(x, f):
    '''
        x: (T,)
        f: (F, 8)
    '''
    padding=len(f)-1 # padding: full
    # reshape input
    x = x.view(1, 1, -1) # 1, C_in, T
    f = torch.flip(f, (0,)).transpose(0, 1).unsqueeze(1) # C_out, C_in, F
    # add reverb
    y = torch.nn.functional.conv1d(x, f, padding=padding)
    y = y.squeeze()[:, :x.shape[2]]
    return y

def batch_resize_noise(clean_reverbs, noise_reverbs):
    N, C, Tc = clean_reverbs.shape
    N, C, Tn = noise_reverbs.shape
    noise_reverbs = noise_reverbs[..., :Tc]
    return noise_reverbs

def batch_rms(reverbs, args):
    # extract parameters
    batch_size = len(reverbs)
    # compute energies
    energies = reverbs ** 2
    max_energies = torch.amax(energies, dim=[1, 2])
    # compute threshold
    thresholds = max_energies * (10 ** (-50 / 10))
    # compute mean rms
    rmss = torch.zeros(batch_size, device=args.device)
    for i in range(len(thresholds)):
        rms     = torch.mean(energies[i, :, :][energies[i, :, :] >= thresholds[i]])
        rmss[i] = torch.maximum(rms, EPS)
    return rmss

def get_scales(clean_rmss, noise_rmss, snr=5):
    return torch.sqrt(clean_rmss / (10 ** (snr / 10) * noise_rmss))

def mix(clean_reverbs, noise_reverbs, args, snr=5, scale=0.8):
    # resize noise reverbs
    # noise_reverbs = batch_resize_noise(clean_reverbs, noise_reverbs)
    # compute rms of noise and clean reverbs
    clean_rmss = batch_rms(clean_reverbs, args)
    noise_rmss = batch_rms(noise_reverbs, args)

    # compute scale to rescale noise according to snr
    scales = get_scales(clean_rmss, noise_rmss, snr=snr)

    # rescale noise
    noise_reverbs *= scales[..., None, None]

    # mix clean and noise
    inputs = clean_reverbs + noise_reverbs
    # 
    max_amplitudes = torch.amax(torch.abs(inputs), dim=[1, 2]) 
    max_amplitudes = torch.maximum(max_amplitudes, torch.full_like(max_amplitudes, EPS))
    scales = 1. / max_amplitudes * scale
    clean_reverbs *= scales[..., None, None]
    inputs        *= scales[..., None, None]
    return inputs, clean_reverbs, noise_reverbs
