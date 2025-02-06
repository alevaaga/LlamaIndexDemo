import onnxruntime as ort
from onnxruntime.capi import _pybind_state as C

print(f"Available ONNXRT providers: {C.get_available_providers()}")

model_path = "/root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel"

providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
ort_sess = ort.InferenceSession(model_path, providers=providers)
