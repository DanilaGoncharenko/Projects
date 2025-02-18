import bentoml
import onnx
import 
# Загрузка модели ONNX
onnx_model = onnx.load('/wine_onnx.onnx')

# Сохранение модели в BentoML
model_info = bentoml.onnx.save_model('onnx_model', onnx_model)
print(f"Model saved: {model_info}")

