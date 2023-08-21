import torch
import onnx
from onnx import version_converter
from torchinfo import summary
from onnx2torch import convert
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node, get_shape_from_value_info
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx, OnnxToTorchModuleWithCustomExport

target_version = 14

# class OnnxLSTM(torch.nn.Module, OnnxToTorchModuleWithCustomExport):
    # def __init__(self):
        # super().__init__()


    # def forward(self, input_tensor: torch.Tensor, W, R, B, h, c) ->torch.Tensor:
        # print('lstm', input_tensor.shape, W.shape, R.shape, B.shape, h.shape, c.shape)
        # return input_tensor

# def get_weights(node: OnnxNode, graph: OnnxGraph):
    # pass

# def get_node(name: str, graph: OnnxGraph):
    # pass



# @add_converter(operation_type='LSTM', version=14)
# def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    # print('lstm input', node.input_values)
    # W_name = node.input_values[1]
    # W_info = graph.value_info[W_name]
    # input_size = get_shape_from_value_info(W_info)[2]
    # hidden_size = node.attributes['hidden_size']

    # print(W_name)
    # print(W_info.type)
    # for node in graph.nodes:
        # print(graph.nodes[node].proto.output[0])
        # if graph.nodes[node].proto.output[0] == W_name:
            # print('find', W_name, 'is', node)
            # print(graph.nodes[node].proto)
    # print(graph.nodes)
    # W = get_weights(get_node(W_name, graph), graph)

    # print('W', W)


    # kwargs = {
        # 'input_size':input_size,
        # 'hidden_size':hidden_size,
    # }
    # module = OnnxLSTM()

    # return OperationConverterResult(
        # torch_module=module,
        # onnx_mapping=onnx_mapping_from_node(node=node),
    # )



path = './models/crnn_lite_lstm'
model = onnx.load(path+'.onnx')
# onnx.save(model, './models/dbnet2.onnx')

converted_model = version_converter.convert_version(model, target_version)
torch_model = convert(converted_model)
torch_model.eval()
# summary(torch_model)
torch.save(torch_model, path+'.pth')

# x = torch.randn(1,3,32,32)
# torch.onnx.export(torch_model, x, './crnn_rev.onnx')
