from pathlib import Path
import yaml


def load_config(path="configs/paths.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    print("CONFIG LOADED:\n")

    for key, value in cfg.items():
        p = Path(value).expanduser()
        print(f"{key}:")
        print(f"  path -> {p}")
        print(f"  exists -> {p.exists()}")
        print()


if __name__ == "__main__":
    main()
