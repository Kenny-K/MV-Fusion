from .model_zoo import PolarOffsetSpconv, PolarOffsetSpconvPytorchMeanshift, PolarOffsetSpconvPytorchFusion

__all__ = {
    'PolarOffsetSpconv': PolarOffsetSpconv,
    'PolarOffsetSpconvPytorchMeanshift': PolarOffsetSpconvPytorchMeanshift,
    'PolarOffsetSpconvPytorchFusion': PolarOffsetSpconvPytorchFusion,
}

def build_network(cfg):
    return __all__[cfg.MODEL.NAME](cfg)
