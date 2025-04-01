from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import pipeline
import torch


def generate_vilt_answer(img, txt):
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    print("loaded model")
    encoding = processor(img, txt, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("predicted answer")
    return model.config.id2label[idx]


def generate_pali_answer(img, txt):
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-vqav2-448")
    model = AutoModelForImageTextToText.from_pretrained("google/paligemma-3b-ft-vqav2-448")

    inputs = processor(images=img, text="<image> " + txt, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(**inputs)

    return processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def generate_llava_answer(img, txt):
    pipe = pipeline("image-to-text", model="llava-hf/llava-1.5-7b-hf")
    prompt = "USER: <image>\n"+txt+"\nASSISTANT:"

    return pipe(img, prompt=prompt)


def generate_blip_answer(img, txt):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")

    inputs = processor(img, txt, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)