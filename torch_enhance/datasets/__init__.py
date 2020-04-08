from .base import BaseDataset
from .bsds300 import BSDS300
from .bsds500 import BSDS500
from .bsds200 import BSDS200
from .bsds100 import BSDS100
from .set5 import Set5
from .set14 import Set14
from .t91 import T91
from .historical import Historical
from .urban100 import Urban100
from .manga109 import Manga109
from .general100 import General100

__all__ = [
    'BaseDataset',
    'BSDS300',
    'BSDS500',
    'BSDS200',
    'BSDS100',
    'Set5',
    'Set14',
    'T91',
    'Historical',
    'Urban100',
    'Manga109',
    'General100'
]