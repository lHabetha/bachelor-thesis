# Scripts

Small helpers for launching the thesis viewers. All paths are relative to the
`bachelor-thesis/` repository root.

## Viewer launchers

| Script | Viewer | Default URL |
|--------|--------|-------------|
| [`view_ch5.sh`](view_ch5.sh) | Chapter 5 clevis optimization | <http://127.0.0.1:8090> |
| [`view_ch6.sh`](view_ch6.sh) | Chapter 6 overlap repair | <http://127.0.0.1:8091> |
| [`view_ch7.sh`](view_ch7.sh) | Chapter 7 AeroForge repair | <http://127.0.0.1:8092> |

Run from anywhere:

```bash
./scripts/view_ch5.sh
./scripts/view_ch6.sh
./scripts/view_ch7.sh
```

Each script `cd`s to the repo root, prepends `code/` to `PYTHONPATH`, and starts
the matching `python -m viewers.*.server` module. Pass extra server flags after
the script name, for example:

```bash
./scripts/view_ch5.sh --port 8099
```

See [`../docs/environment.md`](../docs/environment.md) for Python/CadQuery
dependencies and [`../viewers/README.md`](../viewers/README.md) for viewer scope.
