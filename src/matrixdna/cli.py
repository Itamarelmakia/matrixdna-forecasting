import argparse, yaml
from . import preprocessing, eda, forecasting

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["clean","eda","model","all"])
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    if args.cmd in ("clean","all"):
        preprocessing.run(cfg)
    if args.cmd in ("eda","all"):
        eda.run(cfg)
    if args.cmd in ("model","all"):
        forecasting.run(cfg)

if __name__ == "__main__":
    main()
