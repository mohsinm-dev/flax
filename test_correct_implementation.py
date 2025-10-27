#!/usr/bin/env python3

"""Test implementation following Codex's architecturally sound approach."""

import functools
import inspect
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.transforms import general
from flax.nnx.transforms.transforms import resolve_kwargs
from flax.nnx import graph

def _maybe_unbind_method(f):
    """
    Detect and unbind bound module methods.
    
    Returns:
        (unbound_func, bound_self) if f is a bound module method
        (f, None) otherwise
    """
    # Check if it's a bound method of an NNX Module
    if (inspect.ismethod(f) and 
        hasattr(f, '__self__') and 
        isinstance(f.__self__, nnx.Module)):
        return f.__func__, f.__self__
    return f, None


def remat_corrected(
    f,
    *,
    prevent_cse: bool = True,
    static_argnums = (),
    policy = None,
):
    """Architecturally correct remat implementation."""
    
    # Step 1: Detect and unbind if it's a bound method
    unbound_func, bound_self = _maybe_unbind_method(f)
    
    # Step 2: Build the inner lifted callable (same as original)
    inner_transform = resolve_kwargs()(
        graph.update_context('remat')(
            general.split_inputs(
                jax.checkpoint(
                    general.merge_inputs(unbound_func, ctxtag='remat'),
                    prevent_cse=prevent_cse,
                    static_argnums=static_argnums,
                    policy=policy,
                ),
                ctxtag='remat',
            ),
        )
    )
    
    # Step 3: Return appropriate wrapper
    if bound_self is not None:
        # Bound method case: inject bound_self as first argument
        @functools.wraps(f)
        def bound_method_wrapper(*args, **kwargs):
            return inner_transform(bound_self, *args, **kwargs)
        return bound_method_wrapper
    else:
        # Regular function case: return as-is
        return inner_transform


# Test the corrected implementation
class Model(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(16, 32, rngs=rngs)
        self.x_max = jnp.array(0.0)

    def __call__(self, x):
        # This should now work with the corrected implementation
        return remat_corrected(self.block)(x)

    def block(self, x):
        self.x_max = jnp.maximum(self.x_max, x.max())
        return self.linear(x)


def test_corrected_implementation():
    """Test the architecturally correct implementation."""
    print("Testing architecturally correct remat implementation...")
    
    model = Model(rngs=nnx.Rngs(0))
    x = jnp.ones((3, 16))
    
    # Test with bound method - should work now
    try:
        result = model(x)
        print(f"✓ Bound method with corrected remat worked! Result shape: {result.shape}")
        print(f"✓ x_max was updated: {model.x_max}")
        return True
    except Exception as e:
        print(f"✗ Bound method with corrected remat failed: {e}")
        return False


def test_unbound_still_works():
    """Test that unbound functions still work."""
    print("\nTesting that regular functions still work...")
    
    def regular_func(x):
        return x * 2
    
    try:
        remat_regular = remat_corrected(regular_func)
        result = remat_regular(jnp.array(5.0))
        print(f"✓ Regular function with corrected remat worked: {result}")
        return True
    except Exception as e:
        print(f"✗ Regular function with corrected remat failed: {e}")
        return False


def test_unbinding_detection():
    """Test the unbinding detection logic."""
    print("\nTesting unbinding detection...")
    
    model = Model(rngs=nnx.Rngs(0))
    
    # Test bound method detection
    bound_method = model.block
    unbound_func, bound_self = _maybe_unbind_method(bound_method)
    
    print(f"Bound method detection:")
    print(f"  Original: {bound_method}")
    print(f"  Unbound func: {unbound_func}")
    print(f"  Bound self type: {type(bound_self) if bound_self else None}")
    print(f"  Correctly detected: {bound_self is not None}")
    
    # Test regular function detection
    def regular_func(x):
        return x
    
    unbound_func2, bound_self2 = _maybe_unbind_method(regular_func)
    print(f"\nRegular function detection:")
    print(f"  Original: {regular_func}")
    print(f"  Unbound func: {unbound_func2}")
    print(f"  Bound self: {bound_self2}")
    print(f"  Correctly ignored: {bound_self2 is None}")


if __name__ == "__main__":
    test_unbinding_detection()
    success1 = test_corrected_implementation()
    success2 = test_unbound_still_works()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Codex's approach is architecturally sound.")
    else:
        print("\n❌ Some tests failed.")