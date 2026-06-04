"""Configuration management using OmegaConf."""
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> DictConfig:
    """Load and merge base + experiment config."""
    config_dir = Path(config_path).parent
    experiment_cfg = OmegaConf.load(config_path)

    # Load base config if referenced
    if "defaults" in experiment_cfg:
        base_configs = []
        for default in experiment_cfg.defaults:
            base_path = config_dir / f"{default}.yaml"
            if base_path.exists():
                base_configs.append(OmegaConf.load(base_path))
        # Remove defaults key before merge
        experiment_cfg = OmegaConf.masked_copy(
            experiment_cfg,
            [k for k in experiment_cfg if k != "defaults"]
        )
        # Merge: base first, then experiment overrides
        merged = OmegaConf.merge(*base_configs, experiment_cfg)
        return merged

    return experiment_cfg


def save_config(cfg: DictConfig, save_path: str) -> None:
    """Save resolved config for reproducibility."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_path)
