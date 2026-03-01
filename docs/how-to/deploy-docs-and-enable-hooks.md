# How-to: Deploy Docs and Enable Hooks

This guide configures two automation paths:

- GitHub Pages deployment for MkDocs
- local git hook that runs tests before push

## Deploy MkDocs on GitHub Pages

This repository includes:

- `.github/workflows/deploy-docs.yml` for Pages deployment from `main`
- `.github/workflows/ci.yml` for test+docs CI checks

To enable Pages deployment in GitHub:

1. Open repository **Settings** -> **Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Push to `main` (or run the workflow manually from Actions).

The docs site is built with `mkdocs build --strict` and deployed via
`actions/deploy-pages`.

## Enable Local Test Hook

Install repository-managed hooks:

```bash
./scripts/install_git_hooks.sh
```

This sets `core.hooksPath=.githooks` and activates `.githooks/pre-push`.

On every `git push`, the hook runs:

```bash
python -m pytest -q
```

Push is blocked if tests fail.
