import os
import json


# % celebahq256_cont2 -> Celeb256
# % celeb64_big -> Celeb64
# % afhq256_giant_net_bottlenecked_cont2 -> AFHQ256
# % afhq256_4000 -> AFHQ256Exp1
# % afhq256_4000_bigger_network -> AFHQ256Exp2


# % 1dmodel\_report -> Low1DMix
# % 2dmodel\_dirt\_pits -> Low2DSymMix
# % 2dmodel\_dirt\_pits\_uneven -> Low2DASymMix
# % 2dmodel_density_mapping -> Low2DUnimodal
# % 2dmodel\_density\_matching\_4 -> Low2DBimodal
# % 2dmodel\_s\_curve\_third\_try -> Low2DSCurve

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

basefolder = "logs/training"

for name in name_map:
    args = json.load(open(f"{basefolder}/{name}/args.json"))
    print(args)
    args["log_name"] = name_map[name]
    args["log_dir"] = args["log_dir"].replace(name, name_map[name])
    json.dump(args, open(f"{basefolder}/{name}/args.json", "w"))
    os.rename(f"{basefolder}/{name}", f"{basefolder}/{name_map[name]}")