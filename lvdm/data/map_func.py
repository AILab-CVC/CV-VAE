def frame_select(sample):
    return {"frames": sample["frames"]}


def frame_filter(sample):
    if "frames" in sample:
        return True
    else:
        return False


def latent_filter(sample):
    if "latents" in sample:
        return True
    else:
        return False


def sd_21_select(sample):
    return {"frames": sample["frames"], "caption": sample["caption"]}


def sd_21_filter(sample):
    if "frames" in sample and "caption" in sample:
        return True
    else:
        return False
