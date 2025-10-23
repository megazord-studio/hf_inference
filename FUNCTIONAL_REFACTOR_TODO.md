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

## In Progress 🔄

### Phase 3: Immutable State Management (Priority: High)
- [ ] Make cache in hf_models_service immutable (use frozendict or similar)
- [ ] Review auth middleware for functional approach
- [ ] Document impure functions clearly (I/O, logging)

### Phase 4: Higher-Order Functions (Priority: Medium)
- [ ] Create error handling HOFs/decorators
- [ ] Add function composition utilities
- [ ] Create pipeline functions for request processing

### Phase 5: Runner Edge Cases (Priority: Medium)
- [ ] Review each runner for model-specific edge cases
- [ ] Document assumptions about model inputs/outputs
- [ ] Add validation for model requirements

### Phase 6: Pure Functions Audit (Priority: Low)
- [ ] Audit all functions for side effects
- [ ] Mark impure functions with clear docstrings
- [ ] Separate I/O from pure logic where possible

## Key Improvements Made

1. **Registry is now purely functional**:
   - All operations return new values, never mutate
   - Uses Mapping type for immutability
   - Pure functions: create_registry, register_runner, get_runner, etc.

2. **Route merging completed**:
   - Single /inference endpoint handles both spec and form submissions
   - Pure helper function _parse_spec_from_form (no side effects)
   - Removed .copy() and dict() mutations from runner_spec construction

3. **Improved type safety**:
   - Registry type is Mapping (immutable interface)
   - Clear function signatures showing all inputs/outputs

## Next Steps
1. Make hf_models_service cache immutable
2. Add higher-order functions for error handling
3. Review runners for edge cases
4. Full purity audit
