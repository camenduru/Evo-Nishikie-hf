import random

from PIL import Image, ImageFilter
from controlnet_aux import LineartDetector
from diffusers import EulerDiscreteScheduler
import gradio as gr
import numpy as np
import spaces
import torch

# torch._inductor.config.conv_1x1_as_mm = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.coordinate_descent_check_all_directions = True
from evo_nishikie_v1 import load_evo_nishikie


DESCRIPTION = """# 🐟 Evo-Nishikie
🤗 [モデル一覧](https://huggingface.co/SakanaAI) | 📝 [ブログ](https://sakana.ai/evo-ukiyoe/) | 🐦 [Twitter](https://twitter.com/SakanaAILabs)

[Evo-Nishikie](https://huggingface.co/SakanaAI/Evo-Nishikie-v1)は[Sakana AI](https://sakana.ai/)が教育目的で開発した浮世絵に特化した画像生成モデルです。
入力した単色摺の浮世絵（墨摺絵等）を日本語プロンプトに沿って多色摺の浮世絵（錦絵）風に変換した画像を生成することができます。より詳しくは、上記のブログをご参照ください。
"""
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo may not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_IMAGES_PER_PROMPT = 1
SAFETY_CHECKER = True
if SAFETY_CHECKER:
    from safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    ).to(device)
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    def check_nsfw_images(
        images: list[Image.Image],
    ) -> tuple[list[Image.Image], list[bool]]:
        safety_checker_input = feature_extractor(images, return_tensors="pt").to(device)
        has_nsfw_concepts = safety_checker(
            images=[images], clip_input=safety_checker_input.pixel_values.to(device)
        )

        return images, has_nsfw_concepts


pipe = load_evo_nishikie(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True,
)
# pipe.unet.to(memory_format=torch.channels_last)
# pipe.controlnet.to(memory_format=torch.channels_last)
# pipe.vae.to(memory_format=torch.channels_last)
# # Compile the UNet, ControlNet and VAE.
# pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
# pipe.controlnet = torch.compile(pipe.controlnet, mode="max-autotune", fullgraph=True)
# pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

lineart_detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
image_filter = ImageFilter.MedianFilter(size=3)
BINARY_THRESHOLD = 40


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@spaces.GPU
@torch.inference_mode()
def generate(
    input_image: Image.Image,
    prompt: str,
    seed: int = 0,
    randomize_seed: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    pipe.to(device)

    lineart_image = lineart_detector(input_image, coarse=False, image_resolution=1024)
    lineart_image_filtered = lineart_image.filter(image_filter)
    conditioning_image = lineart_image_filtered.point(lambda p: 255 if p > BINARY_THRESHOLD else 0).convert("L")

    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)

    images = pipe(
        prompt=prompt + "最高品質の輻の浮世絵。超詳細。",
        negative_prompt="暗い",
        image=conditioning_image,
        guidance_scale=7.0,
        controlnet_conditioning_scale=0.8,
        num_inference_steps=35,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        output_type="pil",
    ).images

    if SAFETY_CHECKER:
        images, has_nsfw_concepts = check_nsfw_images(images)
        if any(has_nsfw_concepts):
            gr.Warning("NSFW content detected.")
            return Image.new("RGB", (512, 512), "WHITE"), seed
    return images[0], seed


examples = [
    ["./sample1.jpg", "女性がやかんと鍋を持ち、小屋の前に立っています。背景には室内で会話する人々がいます。"],
    ["./sample2.jpg", "着物を着た女性が、赤ん坊を抱え、もう一人の子どもが手押し車を引いています。背景には木があります。"],
    ["./sample3.jpg", "女性が花柄の着物を着ており、他の人物たちが座りながら会話しています。背景には家の内部があります。"],
    ["./sample4.jpg", "花柄や模様入りの着物を着た男女が室内で集まり、煎茶の準備をしています。背景に木材の装飾があります。"],
]

css = """
.gradio-container{max-width: 1380px !important}
h1{text-align:center}
"""
with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(image_mode="RGB", type="pil", show_label=False)
            prompt = gr.Textbox(placeholder="日本語でプロンプトを入力してください。", show_label=False)
            submit = gr.Button()
            with gr.Accordion("詳細設定", open=False):
                seed = gr.Slider(label="シード値", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="ランダムにシード値を決定", value=True)
        with gr.Column():
            result = gr.Image(label="Evo-Nishikieからの生成結果", type="pil", show_label=False)
    gr.Examples(examples=examples, inputs=[input_image, prompt], outputs=[result, seed], fn=generate)
    gr.on(
        triggers=[
            submit.click,
        ],
        fn=generate,
        inputs=[
            input_image,
            prompt,
            seed,
            randomize_seed,
        ],
        outputs=[result, seed],
        api_name="run",
    )
    gr.Markdown("""⚠️ 本モデルは実験段階のプロトタイプであり、教育および研究開発の目的でのみ提供されています。商用利用や、障害が重大な影響を及ぼす可能性のある環境（ミッションクリティカルな環境）での使用には適していません。
                本モデルの使用は、利用者の自己責任で行われ、その性能や結果については何ら保証されません。
                Sakana AIは、本モデルの使用によって生じた直接的または間接的な損失に対して、結果に関わらず、一切の責任を負いません。
                利用者は、本モデルの使用に伴うリスクを十分に理解し、自身の判断で使用することが必要です。
                アップロードされた画像は画像生成のみに使用され、サーバー上に保存されることはありません。

                出典：サンプル画像はすべて[日本古典籍データセット（国文学研究資料館蔵）『絵本玉かつら』](http://codh.rois.ac.jp/pmjt/book/200013861/)から引用しました。""")

demo.queue().launch()