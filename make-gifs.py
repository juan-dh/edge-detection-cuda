import os
import re
from PIL import Image

outputs_dir = "./data/outputs/"
uscsipi_dir = "./data/uscsipi_images/"
stl10_dir = "./data/stl10_images/"
gif_output = "comparison.gif"

files = sorted([f for f in os.listdir(outputs_dir) if f.lower().endswith(".pgm")])

if not files:
    raise ValueError("No hay archivos .pgm en outputs")

first = files[0]

if first.startswith("uscsipi"):
    ref_dir = uscsipi_dir
elif first.startswith("stl10"):
    ref_dir = stl10_dir
else:
    raise ValueError("Dataset no reconocido")

pairs = []

for name in files[:10]:

    match = re.search(r"img_(\d+)", name)
    if not match:
        continue

    number = int(match.group(1))
    ref_name = f"img_{number:04d}.jpg"

    out_path = os.path.join(outputs_dir, name)
    ref_path = os.path.join(ref_dir, ref_name)

    if not os.path.exists(ref_path):
        continue

    img_out = Image.open(out_path).convert("RGB")
    img_ref = Image.open(ref_path).convert("RGB")

    # igualar altura
    h = min(img_out.height, img_ref.height)

    img_out = img_out.resize((int(img_out.width * h / img_out.height), h))
    img_ref = img_ref.resize((int(img_ref.width * h / img_ref.height), h))

    combined = Image.new("RGB", (img_out.width + img_ref.width, h))
    combined.paste(img_out, (0, 0))
    combined.paste(img_ref, (img_out.width, 0))

    pairs.append(combined)

if not pairs:
    raise ValueError("No se generaron pares")

# ---- crear lienzo del tamaño máximo ----
max_w = max(img.width for img in pairs)
max_h = max(img.height for img in pairs)

frames = []

for img in pairs:
    canvas = Image.new("RGB", (max_w, max_h), (0, 0, 0))
    x = (max_w - img.width) // 2
    y = (max_h - img.height) // 2
    canvas.paste(img, (x, y))
    frames.append(canvas)

frames[0].save(
    gif_output,
    save_all=True,
    append_images=frames[1:],
    duration=800,
    loop=0
)

print("GIF creado:", gif_output)