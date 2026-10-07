from .cli import main

if __name__ == "__main__":   # guard needed on Windows: dataloader workers re-import this module
    raise SystemExit(main())
