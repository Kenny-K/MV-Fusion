from .model_zoo import PolarOffsetSpconv, PolarOffsetSpconvPytorchMeanshift, PolarOffsetSpconvPytorchFusion, PolarOffsetSpconvPytorchFusionCheckPoint

__all__ = {
    'PolarOffsetSpconv': PolarOffsetSpconv,
    'PolarOffsetSpconvPytorchMeanshift': PolarOffsetSpconvPytorchMeanshift,
    'PolarOffsetSpconvPytorchFusion': PolarOffsetSpconvPytorchFusion,
    'PolarOffsetSpconvPytorchFusionCheckPoint': PolarOffsetSpconvPytorchFusionCheckPoint
}

def build_network(cfg):
    return __all__[cfg.MODEL.NAME](cfg)
