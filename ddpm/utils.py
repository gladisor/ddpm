from torch import Tensor

def negative_one_to_one(x: Tensor) -> Tensor:
    return 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0

def viewable(x: Tensor) -> Tensor:
    return ((x + 1.) / 2.).permute(1, 2, 0).clamp(0., 1.).cpu()

def extract(v: Tensor, t: Tensor) -> Tensor:
    '''
    v: vector of values
    t: integer indexes
    '''
    return v.gather(0, t)[:, None, None, None]