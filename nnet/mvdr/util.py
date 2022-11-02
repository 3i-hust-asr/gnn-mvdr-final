from torch_complex.tensor import ComplexTensor
from torch_complex import functional as FC
import torch

def tik_reg(mat, reg: float = 1e-8, eps: float = 1e-8):
    C = mat.size(-1)
    eye = torch.eye(C, dtype=mat.dtype, device=mat.device)
    shape = [1 for _ in range(mat.dim() - 2)] + [C, C]
    eye = eye.view(*shape).repeat(*mat.shape[:-2], 1, 1)
    with torch.no_grad():
        epsilon = FC.trace(mat).real[..., None, None] * reg + eps
    mat = mat + epsilon * eye
    return mat

def get_psd_matrix(spec):
    complex_spec = ComplexTensor(spec[..., 0], spec[..., 1])
    # (Batch, Channel, Frame, Frequency)
    complex_spec = complex_spec.permute(0, 3, 1, 2)
    # (Batch, Frequency, Channel, Frame)
    psd = FC.einsum("...ct,...et->...tce", [complex_spec, complex_spec.conj()])
    # (Batch, Frequency, Frame, Channel, Channel)
    psd = psd.sum(dim=-3)
    # (Batch, Frequency, Channel, Channel)
    return psd

def get_mvdr_weight(psd_speech, psd_noise, reference_vector, eps=1e-8):
    # psd_noise = tik_reg(psd_noise, reg=1e-7, eps=eps)
    # numerator = FC.solve(psd_speech, psd_noise)
    # weight = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # mvdr_weight = FC.einsum("...fec,...c->...fe", [weight, reference_vector])

    numerator = FC.solve(psd_speech, psd_noise)
    weight = numerator / (FC.trace(numerator)[..., None, None])
    mvdr_weight = FC.einsum("...fec,...c->...fe", [weight, reference_vector])
    return mvdr_weight

def apply_mvdr(x, speech_spec, noise_spec, ref_channel=0):
    psd_speech = get_psd_matrix(speech_spec)
    psd_noise = get_psd_matrix(noise_spec)
    # (Batch, Frequency, Channel, Channel)
    data = ComplexTensor(x[..., 0], x[..., 1])
    # (Batch, Frame, Channel, Frequency)
    data = data.permute(0, 3, 2, 1)
    # (Batch, Frequency, Channel, Frame)
    u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)), device=data.device, dtype=torch.float32)
    u[..., ref_channel].fill_(1)

    weight = get_mvdr_weight(psd_speech, psd_noise, u)
    # (Batch, Frequency, Channel)
    enhance = FC.einsum("...c,...ct->...t", [weight.conj(), data])
    # (Batch, Frequency, Frame)
    return enhance, weight

