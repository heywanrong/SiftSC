# Security policy

## Supported versions

The latest release on the default branch receives security fixes.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not open a public issue for a vulnerability that could expose model endpoints, local files, secrets, or user prompts.

SiftSC runs model-generated text through user-configurable parsers and backends. Treat prompts and completions as untrusted data, keep remote backend credentials outside configuration files, and do not expose an unauthenticated SiftSC service to a network.
