import models
import requests
from PIL import Image
from huggingface_hub import login



# 예제 이미지 텍스트
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "How many cats are there?"

answers = [models.generate_vilt_answer(image, text), models.generate_llava_answer(image, text),
           models.generate_blip_answer(image, text), models.generate_pali_answer(image, text)]
print(answers)



