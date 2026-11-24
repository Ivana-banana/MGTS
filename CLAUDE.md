# Team rules for Claude Code

Follow the Windows PowerShell startup checks and working directory in
`AGENTS.md`. The current assignments and file owners are listed in
`COORDINATION.md`.

- Test mode: read the `Tests:` line in `CLAUDE.local.md`. `off`: do not write or run
  tests; check with build, typecheck, lint or screenshot. `ask`: ask before you write
  or run tests. `on` or no line: write tests for logic where an error is not visible
  by eye, and run the smallest test set that covers the change. Change it with `/tests`.
- Never make a check pass by weakening it: do not skip, delete or loosen tests or
  lint rules. Fix the code, or tell me why the check is wrong.
- Before a demo or a submission, offer to run the `silent-failure-hunter` agent.
