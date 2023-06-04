import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=3, kernel_size=(3,3), out_channels=3, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=3, kernel_size=(3,3), out_channels=3, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(2,2))
        self.linear = nn.Linear(bias=True, in_features=12, out_features=4)

        archive = zipfile.ZipFile('demo/test_net.pnnx.bin', 'r')
        self.conv1.bias = self.load_pnnx_bin_as_parameter(archive, 'conv1.bias', (3), 'float32')
        self.conv1.weight = self.load_pnnx_bin_as_parameter(archive, 'conv1.weight', (3,3,3,3), 'float32')
        self.conv2.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2.bias', (3), 'float32')
        self.conv2.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2.weight', (3,3,3,3), 'float32')
        self.linear.bias = self.load_pnnx_bin_as_parameter(archive, 'linear.bias', (4), 'float32')
        self.linear.weight = self.load_pnnx_bin_as_parameter(archive, 'linear.weight', (4,12), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        _, tmppath = tempfile.mkstemp()
        tmpf = open(tmppath, 'wb')
        with archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        tmpf.close()
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.conv1(v_0)
        v_2 = self.conv2(v_1)
        v_3 = (v_2 + v_0)
        v_4 = self.relu(v_3)
        v_5 = self.avgpool(v_4)
        v_6 = torch.flatten(input=v_5, end_dim=-1, start_dim=1)
        v_7 = self.linear(v_6)
        return v_7

def export_torchscript():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 4, 4, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("demo/test_net_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 4, 4, dtype=torch.float)

    torch.onnx._export(net, v_0, "demo/test_net_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def test_inference():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 4, 4, dtype=torch.float)

    return net(v_0)
