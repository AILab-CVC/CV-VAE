from .encoders.modules import GeneralConditioner, JointUCGConditionerForSVD

UNCONDITIONAL_CONFIG = {
    "target": "lvdm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
