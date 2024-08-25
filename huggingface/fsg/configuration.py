from transformers.configuration_utils import PretrainedConfig


class FeConfig:
    def __init__(
        self,
        patch_size: int = 128,
        variant: str = "p128",
        num_classes: int = 0,
        num_filters: int = 6,
        is_constrained: bool = False,
    ):
        self.patch_size = patch_size
        self.variant = variant
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.is_constrained = is_constrained

    def to_dict(self):
        return {
            "patch_size": self.patch_size,
            "variant": self.variant,
            "num_classes": self.num_classes,
            "num_filters": self.num_filters,
            "is_constrained": self.is_constrained,
        }


class CompareNetConfig:
    def __init__(
        self,
        hidden_dim: int = 2048,
        output_dim: int = 64,
    ):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def to_dict(self):
        return {
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }


class FsgConfig(PretrainedConfig):
    model_type = "fsg"
    
    def __init__(
        self,
        fe_config=None,
        comparenet_config=None,
        fast_sim_mode: bool = True,
        loc_threshold: float = 0.3,
        stride_ratio: float = 0.5,
        need_input_255: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fe_config = FeConfig() if fe_config is None else FeConfig(**fe_config)
        self.comparenet_config = CompareNetConfig() if comparenet_config is None else CompareNetConfig(**comparenet_config)
        self.fast_sim_mode = fast_sim_mode
        self.loc_threshold = loc_threshold
        self.stride_ratio = stride_ratio
        self.need_input_255 = need_input_255

    def to_dict(self):
        return {
            "fe_config": self.fe_config.to_dict(),
            "comparenet_config": self.comparenet_config.to_dict(),
            "fast_sim_mode": self.fast_sim_mode,
            "loc_threshold": self.loc_threshold,
            "stride_ratio": self.stride_ratio,
            "need_input_255": self.need_input_255,
        }