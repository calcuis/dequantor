import torch
from transformers import T5EncoderModel
from dequantor import FluxPipeline, GGUFQuantizationConfig, FluxTransformer2DModel

model_path = "https://huggingface.co/calcuis/krea-gguf/blob/main/flux1-krea-dev-iq4_nl.gguf"
transformer = FluxTransformer2DModel.from_single_file(
    model_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    config="callgg/krea-decoder",
    subfolder="transformer"
)

text_encoder = T5EncoderModel.from_pretrained(
    "chatpig/t5-v1_1-xxl-encoder-fp32-gguf",
    gguf_file="t5xxl-encoder-fp32-q2_k.gguf",
    torch_dtype=torch.bfloat16
)

pipe = FluxPipeline.from_pretrained(
    "callgg/krea-decoder",
    transformer=transformer,
    text_encoder_2=text_encoder,
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

prompt = "a pig holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=2.5,
).images[0]
image.save("output.png")
