import bentoml
from bentoml.io import NumpyNdarray, JSON
import numpy as np
svc = bentoml.Service("onnx_service")
onnx_runner = bentoml.onnx.get("onnx_model:latest").to_runner()
svc.runners.append(onnx_runner)

@svc.api(input=NumpyNdarray(), output=JSON())
async def predict_numpy(input_data: np.ndarray):
    try:
        if not isinstance(input_data, np.ndarray):
            return {"error": "Входные данные должны быть массивом NumPy"}, 400
        input_data = np.array(input_data, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        result = await onnx_runner.async_run(input_data)
        predicted_class = int(np.argmax(result[0])) + 1
        return {"predicted_class": predicted_class}
    except Exception as e:
        return {"error": f"Произошла ошибка: {str(e)}"}, 500
   
    