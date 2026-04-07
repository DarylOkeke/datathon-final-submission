from __future__ import annotations

from src import pipeline


def main() -> None:
    print("Step 1: loading and preprocessing raw data...")
    daily_all, interval_all, anchors, cv_training = pipeline.save_preprocessed_data()
    print(f"  daily rows: {len(daily_all):,}")
    print(f"  interval rows: {len(interval_all):,}")
    print(f"  August anchor rows: {len(anchors):,}")
    print(f"  CV training rows: {len(cv_training):,}")

    print("Step 2: training CV interval shape model...")
    _cv_artifacts, cv_forecast = pipeline.run_cv_pipeline()
    print(f"  CV interval forecast rows: {len(cv_forecast):,}")

    print("Step 3: building CCT and abandonment profiles...")
    _cct_profiles, _cct_scale, _abd_profiles, service_forecast = pipeline.run_service_pipeline()
    print(f"  service interval forecast rows: {len(service_forecast):,}")

    print("Step 4: constructing final submission...")
    submission = pipeline.run_final_submission(apply_bias=True)
    pipeline.run_final_submission(apply_bias=False)
    print(f"  final submission rows: {len(submission):,}")
    print(f"Saved: {pipeline.OUTPUT_DIR / 'forecast_v42.csv'}")
    print(f"Saved: {pipeline.OUTPUT_DIR / 'forecast_v42_pre_bias.csv'}")


if __name__ == "__main__":
    main()
