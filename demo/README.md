# GeoWarp Demo Outputs

The scripts under this directory help validate the Warp backend by producing
ParaView-ready snapshots for the representative regression scenarios. Run
`python demo/generate_visualizations.py --arch cpu` to populate `demo/outputs`
with `.vtu`/`.vts` files that mirror the GeoTaichi examples as well as quick
SVG previews for a glanceable inspection. When the Warp runtime is not
available (for instance on documentation builds or when iterating on a laptop
without GPU drivers) the script automatically falls back to generating a
deterministic stub dataset so this folder always contains ready-to-view files.

Each subdirectory corresponds to a solver family:

- `dem/` contains point-cloud VTK exports for the discrete element column
  collapse scenario.
- `mpm/` records particle states for the explicit MPM column collapse test.
- `mpdem/` houses both MPM particles (`mpm/`) and DEM grains (`dem/`) from the
  coupled box sinking setup.

You can open the generated files in ParaView to confirm that the Warp port
preserves the expected kinematics and field names. Adjust the `--frame-interval`
argument to capture additional frames or disable specific scenarios via the
`--scenarios` flag. Pass `--stub` to regenerate the deterministic placeholder
outputs even on machines with Warp installed.
