import os
import shutil

basefolder = "../../U9/Fordypningsprosjekt/diffusion-project/src"
print(os.listdir(basefolder))

files_to_copy = ["denoiser.py", "trainer.py", "utils.py", "ema.py", "run_training.py"]

folders_to_copy = ["models", "data_manager", "custom_types"]

for file in files_to_copy:
    os.system(f"cp {basefolder}/{file} src/")

for folder in folders_to_copy:
    os.system(f"cp -r {basefolder}/{folder} src/")

if os.path.exists("src/experiments"):
    shutil.rmtree("src/experiments")
os.mkdir("src/experiments")

experiments = ["shortest_distance", "density_mapping", "identical_initial_latents", "latent_manipulation"]
experiments = [e + ".ipynb" for e in experiments]

for experiment in experiments:
    os.system(f"cp {basefolder}/experiments/{experiment} src/experiments/")

name_map = {
    "celebahq256_cont2": "Celeb256",
    "celeb64_big": "Celeb64",
    "afhq256_giant_net_bottlenecked_cont2": "AFHQ256",
    "afhq256_4000": "AFHQ256Exp1",
    "afhq256_4000_bigger_network": "AFHQ256Exp2",
    "1dmodel_report": "Low1DMix",
    "2dmodel_dirt_pits": "Low2DSymMix",
    "2dmodel_dirt_pits_uneven": "Low2DASymMix",
    "2dmodel_density_matching": "Low2DUnimodal",
    "2dmodel_density_matching_4": "Low2DBimodal",
    "2dmodel_s_curve_third_try": "Low2DSCurve",
}

# # Scan experiment files and replace names
for experiment in experiments:
    with open(f"src/experiments/{experiment}", "r", encoding="utf-8") as f:
        text = f.read()
    for name in name_map:
        text = text.replace(name, name_map[name])
    # Find where it says "fig.savefig..." and replace with "# fig.savefig..."
    text = text.replace("fig.savefig", "# fig.savefig")
    with open(f"src/experiments/{experiment}", "w", encoding="utf-8") as f:
        f.write(text)



