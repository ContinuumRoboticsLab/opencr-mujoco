"""Configuration loader utilities for opencr-mujoco."""

import copy
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

# Get the project root directory (two levels up from this file: opencr_mujoco/utils/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class ConfigLoader:
    """Load and manage configuration files."""

    def __init__(self, config_dir: str = "configs"):
        """Initialize config loader.

        Args:
            config_dir: Base directory for config files (relative to project root)
        """
        # Make config_dir relative to project root
        config_path = Path(config_dir)
        if not config_path.is_absolute():
            self.config_dir = PROJECT_ROOT / config_path
        else:
            self.config_dir = config_path

    def load_config(self, config_type: str, config_name: str) -> Dict[str, Any]:
        """Load a configuration file.

        Args:
            config_type: Type of config (viewer, teleop, evaluation)
            config_name: Name of config file (without .json extension)

        Returns:
            Dictionary containing configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid
        """
        config_path = self.config_dir / config_type / f"{config_name}.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # Add metadata
        config["_config_path"] = str(config_path)
        config["_config_type"] = config_type
        config["_config_name"] = config_name

        return config

    def list_configs(self, config_type: str) -> list[str]:
        """List available configs for a given type.

        Args:
            config_type: Type of config (viewer, teleop, evaluation)

        Returns:
            List of config names (without .json extension)
        """
        type_dir = self.config_dir / config_type
        if not type_dir.exists():
            return []

        configs = []
        for config_file in type_dir.glob("*.json"):
            configs.append(config_file.stem)

        return sorted(configs)

    def save_config(
        self,
        config: Dict[str, Any],
        config_type: str,
        config_name: str,
        overwrite: bool = False,
    ) -> Path:
        """Save a configuration to file.

        Args:
            config: Configuration dictionary to save
            config_type: Type of config (viewer, teleop, evaluation)
            config_name: Name for config file (without .json extension)
            overwrite: Whether to overwrite existing config

        Returns:
            Path to saved config file

        Raises:
            FileExistsError: If file exists and overwrite=False
        """
        # Ensure directory exists
        type_dir = self.config_dir / config_type
        type_dir.mkdir(parents=True, exist_ok=True)

        config_path = type_dir / f"{config_name}.json"

        if config_path.exists() and not overwrite:
            raise FileExistsError(f"Config already exists: {config_path}")

        # Remove metadata before saving
        config_to_save = {k: v for k, v in config.items() if not k.startswith("_")}

        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        return config_path


def add_config_args(
    parser: argparse.ArgumentParser,
    config_type: str,
    default_config: Optional[str] = None,
):
    """Add standard config arguments to argument parser.

    Args:
        parser: Argument parser to add arguments to
        config_type: Type of config (viewer, teleop, evaluation)
        default_config: Default config name if not specified
    """
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=default_config,
        help=f"Config name to load from configs/{config_type}/",
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help=f"List available {config_type} configs and exit",
    )

    parser.add_argument(
        "--save-config",
        type=str,
        metavar="NAME",
        help="Save current configuration with given name",
    )

    parser.add_argument(
        "--show-config", action="store_true", help="Print loaded configuration and exit"
    )


def handle_config_args(
    args: argparse.Namespace, config_type: str, default_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Handle config-related arguments and load configuration.

    Args:
        args: Parsed command line arguments
        config_type: Type of config (viewer, teleop, evaluation)
        default_config: Default configuration if none specified

    Returns:
        Loaded configuration dictionary
    """
    loader = ConfigLoader()

    # List configs if requested
    if args.list_configs:
        configs = loader.list_configs(config_type)
        if configs:
            print(f"Available {config_type} configs:")
            for config in configs:
                print(f"  - {config}")
        else:
            print(f"No {config_type} configs found in configs/{config_type}/")
        sys.exit(0)

    # Load config
    if args.config:
        try:
            config = loader.load_config(config_type, args.config)
            print(f"Loaded config: {args.config}")
        except FileNotFoundError:
            print(f"Config not found: {args.config}")
            print(f"Available configs: {', '.join(loader.list_configs(config_type))}")
            sys.exit(1)
    else:
        config = copy.deepcopy(default_config)
        print("Using default configuration")

    # Override config with command line args: CLI args > config file > default.
    # Only values the user actually provided are merged. This requires every
    # config-overridable argparse option to default to None (boolean flags use
    # action="store_const", const=True, default=None — a plain store_true
    # default of False would silently clobber config-file values).
    for key, value in vars(args).items():
        if key not in ["config", "list_configs", "save_config", "show_config"]:
            if value is not None:
                config[key] = value

    # Show config if requested
    if args.show_config:
        print(f"\nConfiguration ({config_type}):")
        for key, value in sorted(config.items()):
            if not key.startswith("_"):
                print(f"  {key}: {value}")
        sys.exit(0)

    # Save config if requested
    if args.save_config:
        try:
            path = loader.save_config(config, config_type, args.save_config)
            print(f"Configuration saved as '{args.save_config}'")
            print(f"Saved to: {path}")
            sys.exit(0)
        except FileExistsError:
            print(f"Config already exists: {args.save_config}")
            print("Use a different name or delete the existing config")
            sys.exit(1)

    return config
