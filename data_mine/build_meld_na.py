"""Build admission-local MELD-Na trajectories; no private mechanism is run."""
import argparse
import json
from data_mine.finite_trajectories import DEFAULT_OUT, DEFAULT_SOURCE, build


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', default=str(DEFAULT_OUT / 'meld_only'))
    p.add_argument('--source', default=str(DEFAULT_SOURCE))
    p.add_argument('--turns', type=int, default=40)
    p.add_argument('--min-updates', type=int, default=16)
    p.add_argument('--max-updates', type=int, default=24)
    p.add_argument('--patients', type=int, default=20)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    print(json.dumps(build(a.output, a.source, ['MELD_NA'], a.turns, a.min_updates,
                           a.max_updates, a.patients, a.overwrite), indent=2))


if __name__ == '__main__':
    main()
