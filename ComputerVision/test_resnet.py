import pytest
import numpy.testing as npt
import numpy as np
from ResNet import IdentityBlock, ConvBlock, ResNet


@pytest.fixture(scope='module', name='identity_block')
def identity_block_fixture():
    return IdentityBlock([2, 2, 2], 3)


@pytest.fixture(scope='module', name='conv_block')
def conv_block_fixture():
    return ConvBlock([2, 2, 2], 3)


def test_call_identity_block(identity_block):
    x = np.random.rand(1, 3, 3, 2)
    y = identity_block(x)
    npt.assert_array_equal(x.shape, y.shape)


def test_call_conv_block(conv_block):
    x = np.random.rand(1, 3, 3, 2)
    y = conv_block(x)
    npt.assert_array_equal(x.shape, y.shape)


def test_resnet():
    x = np.random.rand(1, 3, 3, 2)
    model = ResNet()
    y = model(x)