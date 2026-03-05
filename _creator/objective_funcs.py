import torch


# internal use only
def _l1_loss(pred, l1_factor):
    if l1_factor == 0: return 0.0
    return l1_factor * torch.abs(pred).mean()


# internal use only
def _tv_loss(imgs, tv_factor): # sum across image dimensions, mean across batch dimension
    tv_x_diff = torch.abs(imgs[..., 1:, :] - imgs[..., :-1, :])
    tv_x_diff = torch.mean(torch.sum(tv_x_diff, dim=tv_x_diff.shape[1:]), dim=0)
    tv_y_diff = torch.abs(imgs[..., :, 1:] - imgs[..., :, :-1])
    tv_y_diff = torch.mean(torch.sum(tv_y_diff, dim=tv_y_diff.shape[1:]), dim=0)
    return tv_factor * (tv_x_diff + tv_y_diff)


def direction_neuron_cossim(model, get_activations, tv_factor=0, l1_factor=0, cossim_pow=4, **kwargs):
    use_tv_loss = tv_factor != 0
    pred, target, imgs = get_activations(model, get_imgs=use_tv_loss)
    l1_loss = _l1_loss(pred, l1_factor)
    tv_loss = _tv_loss(imgs, tv_factor) if use_tv_loss else 0.0
    mag = torch.sqrt(torch.sum(pred ** 2, dim=1))  # [batch_size, y, x]
    dot = torch.mean(pred * target, dim=1)  # [batch_size, y, x]
    cossim = dot / (mag + 1e-6)
    compare_tensor = (torch.zeros_like(cossim) + 0.1).to(cossim.device)
    cossim = torch.maximum(compare_tensor, cossim)
    result_per_elem = -dot * (cossim ** cossim_pow)
    return l1_loss + tv_loss + torch.mean(result_per_elem)


def unscaled_cossim(model, get_activations, tv_factor=0, l1_factor=0, **kwargs):
    use_tv_loss = tv_factor != 0
    pred, target, imgs = get_activations(model, get_imgs=use_tv_loss)
    l1_loss = _l1_loss(pred, l1_factor)
    tv_loss = _tv_loss(imgs, tv_factor) if use_tv_loss else 0.0
    mag = torch.sqrt(torch.sum(pred ** 2, dim=1))
    dot = torch.mean(pred * target, dim=1)
    cossim = dot / (mag + 1e-6)
    return l1_loss + tv_loss + torch.mean(-cossim)


def dot(model, get_activations, tv_factor=0, l1_factor=0, **kwargs):
    use_tv_loss = tv_factor != 0
    pred, target, imgs = get_activations(model, get_imgs=use_tv_loss)
    l1_loss = _l1_loss(pred, l1_factor)
    tv_loss = _tv_loss(imgs, tv_factor) if use_tv_loss else 0.0
    dot = (pred * target).sum(dim=1)
    return l1_loss + tv_loss + torch.mean(-dot)


def euclidean(model, get_activations, tv_factor=0, l1_factor=0, **kwargs):
    use_tv_loss = tv_factor != 0
    pred, target, imgs = get_activations(model, get_imgs=use_tv_loss)
    l1_loss = _l1_loss(pred, l1_factor)
    tv_loss = _tv_loss(imgs, tv_factor) if use_tv_loss else 0.0
    return l1_loss + tv_loss - torch.mean(1 - torch.linalg.norm(pred - target, ord=2, dim=1))


# minimize cross entropy for target class
def cross_entropy(model, get_activations, tv_factor=0, l1_factor=0, **kwargs):
    use_tv_loss = tv_factor != 0
    pred, target, imgs = get_activations(model, get_imgs=use_tv_loss)
    l1_loss = _l1_loss(pred, l1_factor)
    tv_loss = _tv_loss(imgs, tv_factor) if use_tv_loss else 0.0
    return l1_loss + tv_loss + torch.nn.functional.cross_entropy(pred, target)


# maximize logit of target class
def max_raw_logit(model, get_activations, tv_factor=0, l1_factor=0, **kwargs):
    use_tv_loss = tv_factor != 0
    # pred: shape [bs, num_cls]
    # target: shape [bs]
    pred, target, imgs = get_activations(model, get_imgs=use_tv_loss)
    bs = pred.shape[0]
    l1_loss = _l1_loss(pred, l1_factor)
    tv_loss = _tv_loss(imgs, tv_factor) if use_tv_loss else 0.0
    logit_loss = -torch.mean(pred[torch.arange(0, bs, step=1), target])
    return l1_loss + tv_loss + logit_loss


# maximize probability of target class
def max_prob(model, get_activations, tv_factor=0, l1_factor=0, **kwargs):
    use_tv_loss = tv_factor != 0
    pred, target, imgs = get_activations(model, get_imgs=use_tv_loss)
    bs = pred.shape[0]
    l1_loss = _l1_loss(pred, l1_factor)
    tv_loss = _tv_loss(imgs, tv_factor) if use_tv_loss else 0.0
    prob_loss = -torch.mean(torch.nn.functional.softmax(pred, dim=1)[torch.arange(0, bs, step=1), target])
    return l1_loss + tv_loss + prob_loss