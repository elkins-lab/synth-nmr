# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.1] - 2026-02-21

### Changed
- **Numpy Compatibility**: Relaxed numpy dependency pin to `<3.0` to resolve binary incompatibilities (`numpy.dtype size changed`) encountered when installed alongside `numpy 2.x`.
