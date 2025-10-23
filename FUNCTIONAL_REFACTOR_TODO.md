# Functional Programming Refactor TODO

## Objectives from @busykoala feedback:
1. Make code more functional than objective
2. Focus on: Purity, Immutability, Disciplined state, First-class functions, Higher-order functions
3. Merge run_form and inference routes (run_form just calls inference)
4. Review runner implementations for edge cases and model coverage

## Completed ✅

### Phase 1: Functional Registry ✅
- [x] Replace `RunnerRegistry` class with functional approach
- [x] Use immutable data structures (frozen dict/Mapping)
- [x] Create pure functions for registry operations
- [x] Update tests to match functional approach (added immutability tests)

### Phase 2: Merge Routes ✅
- [x] Analyze run_form endpoint functionality
- [x] Move form parsing logic to inference endpoint as pure helper
- [x] Support both JSON spec and form-based submission in single endpoint
- [x] Simplify run_form route to delegate to inference
- [x] Remove mutable operations (.copy(), dict() calls)

### Phase 3: Functional Utilities ✅
- [x] Create higher-order functions module (compose, pipe, curry, safe_call)
- [x] Add Result type for functional error handling
- [x] Create runner utilities with HOFs (with_standard_error_handling, etc.)
- [x] Document impure functions clearly (cache mutations)

## In Progress 🔄

### Phase 4: Documentation & Examples
- [x] Document pure vs impure functions
- [x] Add docstrings explaining side effects
- [x] Create example showing functional composition
- [ ] Update more runners with functional pattern examples

### Phase 5: Runner Edge Cases (Priority: Medium)
- [ ] Review each runner for model-specific edge cases
- [ ] Document assumptions about model inputs/outputs
- [ ] Add validation for model requirements
- [ ] Add tests for edge cases

## Key Improvements Made

1. **Registry is now purely functional**:
   - All operations return new values, never mutate
   - Uses Mapping type for immutability
   - Pure functions: create_registry, register_runner, get_runner, etc.

2. **Route merging completed**:
   - Single /inference endpoint handles both spec and form submissions
   - Pure helper function _parse_spec_from_form (no side effects)
   - Removed .copy() and dict() mutations from runner_spec construction

3. **Functional utilities created**:
   - `app/infrastructure/functional.py`: compose, pipe, curry, safe_call, etc.
   - `app/infrastructure/runner_utils.py`: HOFs for runners
   - Higher-order functions for error handling

4. **Clear documentation of purity**:
   - Impure functions explicitly documented (I/O, mutations)
   - Pure functions marked in docstrings
   - Side effects clearly stated

## Functional Programming Patterns Implemented

1. **Pure Functions** ✅
   - Registry operations
   - Form parsing helper
   - All transformation logic

2. **Immutability** ✅
   - Registry returns new values
   - No mutations in core logic
   - Immutable Mapping types

3. **Higher-Order Functions** ✅
   - compose, pipe for function composition
   - with_standard_error_handling for runners
   - validate_spec_fields decorator
   - memoize for caching pure functions

4. **Functional Error Handling** ✅
   - Result type (value, error) tuples
   - safe_call for exception-free error handling
   - either for Result processing

5. **First-Class Functions** ✅
   - Runners passed as values
   - HOFs accept and return functions
   - Currying support

## Architecture Summary

```
Pure Functions (No Side Effects):
- Registry operations (create, register, get, etc.)
- Form parsing (_parse_spec_from_form)
- Data transformations (safe_json, gated_to_str)
- Composition utilities (compose, pipe)

Impure Functions (Clearly Documented):
- Runner execution (calls external APIs)
- Cache operations (module-level state mutations)
- File I/O (disk reads/writes)
- Logging (side effect)
- HTTP requests (network I/O)
```

## Next Steps
1. Review runners for edge cases
2. Add more runner examples with HOFs
3. Consider more functional patterns where beneficial
4. Performance testing
