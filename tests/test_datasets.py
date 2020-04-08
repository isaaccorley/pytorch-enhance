import pytest

from torch_enhance import datasets


SCALE_FACTOR = 2

def test_BSDS300():
    data = datasets.BSDS300(SCALE_FACTOR)

def test_BSDS500():
    data = datasets.BSDS500(SCALE_FACTOR)

def test_BSDS200():
    data = datasets.BSDS200(SCALE_FACTOR)

def test_BSDS100():
    data = datasets.BSDS100(SCALE_FACTOR)

def test_Set5():
    data = datasets.Set5(SCALE_FACTOR)

def test_Set14():
    data = datasets.Set14(SCALE_FACTOR)

def test_T91():
    data = datasets.T91(SCALE_FACTOR)

def test_Historical():
    data = datasets.Historical(SCALE_FACTOR)

def test_General100():
    data = datasets.General100(SCALE_FACTOR)

def test_Urban100():
    data = datasets.Urban100(SCALE_FACTOR)

def test_Manga109():
    data = datasets.Manga109(SCALE_FACTOR)