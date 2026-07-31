"""Build the frozen BTH event manifest and optionally fit local Z--R."""

import argparse
import json

from openstl.datasets.radar_protocol import (
    build_manifest, fit_local_zr, fit_local_zr_from_rain)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--station-csv')
    parser.add_argument('--fit-rain-png', action='store_true')
    parser.add_argument('--zr-output')
    parser.add_argument('--dbz-threshold', type=float, default=20.0)
    parser.add_argument('--wet-fraction', type=float, default=0.01)
    parser.add_argument('--max-dry-gap-hours', type=float, default=3.0)
    parser.add_argument('--padding-minutes', type=int, default=30)
    parser.add_argument('--scan-workers', type=int, default=8)
    parser.add_argument('--start-date', default='2025-05-01')
    parser.add_argument('--end-date', default='2025-08-31')
    parser.add_argument('--train-end', default='2025-07-31T23:59:59')
    args = parser.parse_args()
    manifest = build_manifest(
        args.data_root, args.manifest,
        dbz_threshold=args.dbz_threshold,
        wet_fraction=args.wet_fraction,
        max_dry_gap_hours=args.max_dry_gap_hours,
        padding_minutes=args.padding_minutes,
        scan_workers=args.scan_workers,
        start_date=args.start_date,
        end_date=args.end_date,
        train_end=args.train_end)
    print(json.dumps({'events': len(manifest['events']),
                      'samples': len(manifest['samples'])}, indent=2))
    if (args.station_csv or args.fit_rain_png) and not args.zr_output:
        parser.error('--zr-output is required when fitting Z-R')
    if args.station_csv and args.fit_rain_png:
        parser.error('choose station CSV or Rain PNG calibration, not both')
    if args.fit_rain_png:
        print(json.dumps(fit_local_zr_from_rain(
            args.manifest, args.data_root, args.zr_output), indent=2))
    elif args.station_csv:
        print(json.dumps(fit_local_zr(
            args.manifest, args.station_csv, args.zr_output), indent=2))


if __name__ == '__main__':
    main()
