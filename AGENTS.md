# Project guidance

- Read `AI_CONTEXT.yaml` and `COVERAGE.md` first; verify state against current files.
- Preserve source formulations and decisions. Do not replace missing algorithms
  with generic OpenCV/SLAM methods or claim unverified completeness.
- Keep this directory independent: no runtime paths, symlinks, imports or native
  library links to the original ORB-SLAM3 checkout. Default tests use local goldens.
- Install dependencies in this directory's `.venv/`, a dedicated Conda environment.
- Keep core algorithms separate from inputs, configuration, UI and exports.
- Update AI context, coverage and the short one-line-per-file setup inventory
  after substantive changes. Record unresolved discrepancies honestly.
