from .base import Base
from .baseline import Bicubic
from .srcnn import SRCNN
from .edsr import EDSR
from .vdsr import VDSR
from .espcn import ESPCN
from .srresnet import SRResNet


__all__ = [
    'Bicubic',
    'SRCNN',
    'VDSR',
    'EDSR'
    'ESPCN',
    'SRResNet'
]