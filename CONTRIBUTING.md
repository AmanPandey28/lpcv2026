# Contributing

Keep changes focused on export correctness, retrieval validation, deployment
reproducibility, or measured optimization.

1. Run the unit tests and publication audit described in the README.
2. Add regression coverage for preprocessing, ordering, masks, shapes, dtypes,
   metric definitions, and acceptance gates.
3. Keep device, precision, dataset digest, graph hash, iteration count, and
   measurement boundary with every benchmark claim.
4. Distinguish compile success, runtime execution, numerical fidelity, and task
   accuracy. Do not overwrite historical evidence with a newer result.
5. Do not commit credentials, binaries, datasets, raw logs, personal documents,
   or local machine paths.

A new exported graph needs full local and device validation before being
described as equivalent to the historical submission. CI does not submit cloud
jobs or require access to private AI Hub records.
