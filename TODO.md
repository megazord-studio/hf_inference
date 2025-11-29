# TODO

## General Instructions

All code should be implemented DRY, KISS, clean code.
Take care it has a well design folder structure.
Pick good naming for functions, variables, classes, files, folders, etc.

Try to implement the code functional with no side effects when possible.
Functions should only do one thing and do it well.
File should not become too large. Around 200 lines is a good limit.

Write tests for all the code. The tests should run real models.
It's ok if they take quite a while. Do not skip or mark them as slow.
Use the inference endpoint for the tests using a fastapi test client.

Imports should not be conditional. Assume libraries are installed. No
try/except around imports. Add necessary libraries using uv.

Use the `./frontend` as reference for what needs to be working in general.

Do not use environment variables to enable or disable capabilities; implement
capabilities as either always-on (hardcoded).

Do not add any fallbacks if a model does not run. It should be obvious that
something is not working as expected. Fallback would hide that fact.

## Next steps

- [ ] Current test fixes
- [ ] Implement Comparison Mode
- [ ] Text-to-Text Incl. Internet Search & "Thinking"
- [ ] Dynamic Issue finder by Trending/Download/Like Score
