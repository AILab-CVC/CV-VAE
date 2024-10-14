def svd_filter(batch):
    if (
        "frames" in batch
        and "cond_frames_without_noise" in batch
        and "fps_id" in batch
        and "motion_bucket_id" in batch
        and "cond_frames" in batch
        and "cond_aug" in batch
    ):
        return True
    else:
        return False


def frames_filter(batch):
    if "frames" in batch:
        return True
    else:
        return False

def left_filter(batch):
    if "left" in batch:
        return True
    else:
        return False