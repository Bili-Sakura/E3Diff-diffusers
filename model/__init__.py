import logging
logger = logging.getLogger('base')


def create_model(opt):
    model_type = opt.get('model', {}).get('which_model_G', 'sr3')

    if model_type == 'cut':
        from .cut.cut_model import CUTModel as M
    else:
        from .model import DDPM as M

    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
