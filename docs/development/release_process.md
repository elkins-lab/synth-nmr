# Release Process

This document outlines the standardized process for releasing new versions of `synth-nmr` to PyPI.

## Automated Deployment Script

We use a unified `publish.sh` script to handle the build, test validation, and PyPI upload steps cleanly. This prevents human error and ensures that a release cannot happen if the tests or build steps fail.

### 1. Bump the Version
Before deploying, ensure you have bumped the version number in both `pyproject.toml` and `synth_nmr/__init__.py`.

### 2. Update the Changelog
Log the new features and bug fixes in `CHANGELOG.md` under the new version header.

### 3. Run the Publish Script

You can deploy to the **Test PyPI** environment to verify the build:
```bash
./publish.sh test
```

When you are ready to push to **Production PyPI**:
```bash
./publish.sh prod
```

The script will automatically:
1. Extract the current version from `pyproject.toml`.
2. Run the entire `pytest` suite. **It will immediately abort if any tests fail.**
3. Clean out old `/dist` and `/build` directories.
4. Build the modern `.whl` and `.tar.gz` distributions.
5. Prompt you for a final confirmation before pushing to PyPI.

### 4. Tag the Release
Once published, commit the version bump and tag it on GitHub:
```bash
git add pyproject.toml synth_nmr/__init__.py CHANGELOG.md
git commit -m "chore(release): bump version to vX.Y.Z"
git tag vX.Y.Z
git push origin master
git push origin vX.Y.Z
```
