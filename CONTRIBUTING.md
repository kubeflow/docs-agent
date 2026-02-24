# Kubeflow Contributor Guide

Welcome to the Kubeflow project! We'd love to accept your patches and
contributions to this project. Please read the
[contributor's guide in our docs](https://www.kubeflow.org/docs/about/contributing/).

The contributor's guide:

* Shows you where to find the Developer Certificate of Origin (DCO) that you need
  to agree to
* Helps you get started with your first contribution to Kubeflow
* Describes the pull request and review workflow in detail, including the
  OWNERS files and automated workflow tool

---

## Development Setup

### Pre-commit Hooks

This repository uses `pre-commit` to enforce basic code quality checks such as:

- Trailing whitespace removal
- End-of-file fixes
- Basic formatting
- Import ordering (if applicable)

### Install Pre-commit

```bash
pip install pre-commit
```

Then install the git hooks:

```bash
pre-commit install
```

To run hooks on all files:

```bash
pre-commit run --all-files
```

Note: If hook modify files, make sure to review changes, stage them and commit again!
