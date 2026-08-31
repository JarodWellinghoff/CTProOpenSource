# License decision required

This preparation bundle intentionally does **not** include an active `LICENSE`
file. Publishing code without a license does not grant outside users permission
to copy, modify, or redistribute it.

Before making the repository public:

1. Confirm who owns the code and whether employer, sponsor, collaborator, or
   third-party approvals are required.
2. Confirm whether any source material was translated from another licensed
   implementation.
3. Have the owner approve a license. Common permissive options include
   Apache-2.0, BSD-3-Clause, and MIT; they differ in patent language,
   attribution requirements, and institutional preferences.
4. Add the exact approved license as the repository's top-level `LICENSE` file.
5. Add the matching license identifier to `pyproject.toml` and `CITATION.cff`.
6. Review source-file headers and third-party notices.

For a research project with possible patent considerations, Apache-2.0 is often
worth legal review because it contains an explicit patent grant. This is not
legal advice, and no license should be selected solely from this note.
