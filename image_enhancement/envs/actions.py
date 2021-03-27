import torch


def gamma_corr(image, gamma, channel=None):
    mod = image.clone()
    if channel is not None:
        mod[:, channel, :, :] = torch.clamp(mod[:, channel, :, :] ** gamma, 0, 1)
    else:
        mod = torch.clamp(mod ** gamma, 0, 1)
    return mod


def brightness(image, bright, channel=None):
    mod = image.clone()
    if channel is not None:
        mod[:, channel, :, :] = torch.clamp(mod[:, channel, :, :] + bright, 0, 1)
    else:
        mod = torch.clamp(mod + bright, 0, 1)

    return mod


def contrast(image, alpha, channel=None):
    mod = image.clone()
    if channel is not None:
        mod[:, channel, :, :] = torch.clamp(
            torch.mean(mod[:, channel, :, :]) + alpha * (mod[:, channel, :, :] - torch.mean(mod[:, channel, :, :])),
            0, 1)
    else:
        mod = torch.clamp(torch.mean(mod) + alpha * (mod - torch.mean(mod)), 0, 1)
    return mod