# GitHub release checklist — 0.4.2

- [ ] Push the exact release source tree to the default branch.
- [ ] Confirm `git status` is clean.
- [ ] Run `tests/run_python_tests.sh`.
- [ ] Confirm the final Metal and Vulkan validation results are recorded in `docs/RELEASE_0.4.2.md`.
- [ ] Confirm `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `CREDITS.md` are present at repository root.
- [ ] Create annotated tag `v0.4.2` from the reviewed release commit.
- [ ] Create GitHub Release titled `moge-ggml 0.4.2` from tag `v0.4.2`.
- [ ] Use `docs/GITHUB_RELEASE_0.4.2.md` as the release body (omit its metadata heading/checklist text if desired).
- [ ] Attach `moge-ggml-0.4.2.tar.gz` and `moge-ggml-0.4.2.tar.gz.sha256`.
- [ ] Verify the uploaded tarball checksum matches the published `.sha256` file.
- [ ] Mark as a normal release, not a pre-release.
