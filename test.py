from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model_path = Path.home() / "models/Qwen3-VL-2B-Instruct"
image_path = Path("validation/model-view.png")

assert model_path.is_dir(), f"Model folder missing: {model_path}"
assert image_path.is_file(), f"Image missing: {image_path}"
assert torch.cuda.is_available(), "CUDA is unavailable in this environment."

print("GPU:", torch.cuda.get_device_name(0))
print("Loading model...", flush=True)

processor = AutoProcessor.from_pretrained(
    str(model_path), local_files_only=True
)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    str(model_path),
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    local_files_only=True,
).eval()

image = Image.open(image_path).convert("RGB")
image.thumbnail((512, 512))

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text":
         "Describe the visible robot and objects in two short sentences. "
         "Describe only what you can see."}
    ]
}]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    output = model.generate(
        **inputs, max_new_tokens=96, do_sample=False
    )

answer = processor.decode(
    output[0, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)
print("\nMODEL RESPONSE:\n", answer)

