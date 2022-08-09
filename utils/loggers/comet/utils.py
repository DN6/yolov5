import os
from io import BytesIO

import comet_ml

COMET_PREFIX = "comet://"
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")


def download_model_checkpoint(opt, experiment):
    model_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(model_dir, exist_ok=True)

    model_name = opt.comet_model_name if opt.comet_model_name else COMET_MODEL_NAME
    model_asset_list = sorted(
        experiment.get_model_asset_list(model_name),
        key=lambda x: x["step"],
        reverse=True,
    )
    latest_model = model_asset_list[0]

    asset_id = latest_model["assetId"]
    model_filename = latest_model["fileName"]

    model_binary = BytesIO(
        experiment.get_asset(asset_id, return_type="binary", stream=False)
    )
    model_download_path = f"{model_dir}/{model_filename}"
    with open(model_download_path, "wb") as f:
        f.write(model_binary.getbuffer())

    opt.weights = model_download_path


def set_opt_parameters(opt, parameters_summary):
    for parameter in parameters_summary:
        parameter_name = parameter["name"]
        parameter_value = parameter["valueCurrent"]

        if parameter_name.startswith("opt"):
            opt_key = parameter_name.replace("opt_", "")
            if opt_key not in ["weights", "resume"]:
                setattr(opt, opt_key, parameter_value)

        import ipdb

        ipdb.set_trace()


def check_comet_weights(opt):
    api = comet_ml.API()
    if isinstance(opt.weights, str):
        if opt.weights.startswith(COMET_PREFIX):
            experiment_path = opt.weights.replace(COMET_PREFIX, "")
            experiment = api.get(experiment_path)
            download_model_checkpoint(opt, experiment)
            return True

    return None


def check_comet_resume(opt):
    api = comet_ml.API()

    if isinstance(opt.resume, str):
        if opt.resume.startswith(COMET_PREFIX):
            experiment_path = opt.resume.replace(COMET_PREFIX, "")
            experiment = api.get(experiment_path)
            parameters_summary = experiment.get_parameters_summary()
            set_opt_parameters(opt, parameters_summary)
            download_model_checkpoint(opt, experiment)
            return True

    return None
