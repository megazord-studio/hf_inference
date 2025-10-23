# Phase 4: Cleanup and Optimization TODO

## From @busykoala feedback (comment #3438812975):

### 1. Remove app/routes/run_form.py ✅ HIGH PRIORITY
- [x] Verify run_form.py is still present
- [ ] Check all references to run_form routes
- [ ] Remove the file entirely
- [ ] Update main.py route registration
- [ ] Update any imports

### 2. Side Effects Audit
- [ ] Review all functions for hidden side effects
- [ ] Document any remaining impure functions
- [ ] Ensure no unexpected mutations
- [ ] Check global state access

### 3. Decorator Pattern vs Iterators
- [ ] Review current iterator usage
- [ ] Consider decorator pattern where beneficial
- [ ] Evaluate readability impact
- [ ] Assess performance implications

### 4. Enhanced Caching
- [ ] Increase cache TTL from 10min to 4h
- [ ] Implement file-based cache in /tmp
- [ ] Add timestamp-based cache keys
- [ ] Test cache persistence across runs

### 5. Frozen Data Structures
- [ ] Identify all constants
- [ ] Convert to frozen structures (tuple, frozenset, etc.)
- [ ] Use types.MappingProxyType for dicts
- [ ] Verify immutability

### 6. JavaScript Minimization
- [ ] Review all JS files
- [ ] Remove unnecessary code
- [ ] Improve readability and docs
- [ ] Ensure minimal implementation

### 7. Remove Unused Files
- [ ] Identify top-level re-export files
- [ ] Remove app/helpers.py (re-export only)
- [ ] Remove app/auth.py (re-export only)
- [ ] Remove app/form_schemas.py (re-export only)
- [ ] Remove app/services/__init__.py if empty
- [ ] Update all imports to use direct paths
- [ ] Remove app/routes/*.py re-exports

### 8. Import Refactoring
- [ ] Update all imports to direct module paths
- [ ] Remove circular dependencies
- [ ] Ensure clean import structure
- [ ] Update tests

## Implementation Order
1. Remove run_form.py (quick win)
2. Enhanced caching (immediate benefit)
3. Frozen data structures (safety)
4. Remove unused re-export files (cleanup)
5. Update imports (consistency)
6. Side effects audit (verification)
7. JS minimization (polish)
8. Decorator review (optimization)
