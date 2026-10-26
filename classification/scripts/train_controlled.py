"""Run one controlled Stage 2 fold; all data/output roots are configurable."""
import argparse
from dataclasses import fields
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.trainer import ControlledConfig, Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in fields(ControlledConfig):
        option = '--'+field.name.replace('_', '-')
        if field.type is bool:
            parser.add_argument(option, action='store_true')
        else:
            parser.add_argument(option, type=field.type, default=field.default)
    parser.add_argument('--resume', nargs='?', const='auto')
    parser.add_argument('--stop-after', type=int, help='Pause after this epoch without changing the planned LR schedule')
    args = vars(parser.parse_args())
    resume, stop = args.pop('resume'), args.pop('stop_after')
    Trainer(ControlledConfig(**args), resume=resume).fit(stop_after=stop)


if __name__ == '__main__':
    main()
