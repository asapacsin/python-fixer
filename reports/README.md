# Research reporting policy

No experiments have run. This directory currently contains no performance results.

Future reports must cite immutable run/data/model/config hashes, exact denominators,
real hardware and measured costs; show negative and interrupted runs; distinguish
validation from heldout; and include uncertainty/contamination limits. Large logs,
weights, datasets, private tests and candidate payloads remain in controlled
external storage. Commit only permitted summaries, analysis code and artifact
manifests without secrets or private test details.

M0 CPU checks are specification consistency evidence. They are not a sandbox,
CUDA, optimizer, trained model, independent reviewer or hosted CI result.
