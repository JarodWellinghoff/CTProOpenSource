# Security and privacy

## Supported versions

Security fixes are applied to the latest released version.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting feature after it is enabled for
the repository. Do not open a public issue for a suspected vulnerability,
credential exposure, internal network path, or data-governance incident.

## Medical-image privacy

DICOM metadata can contain protected health information. File extensions,
header removal, and the `.gitignore` rules in this repository are not
de-identification. Follow an institutionally approved de-identification
workflow and verify the resulting files before any upload.

Do not include clinical data in bug reports. Prefer generated arrays or a
purpose-built, non-identifiable phantom dataset.
