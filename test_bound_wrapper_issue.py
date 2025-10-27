#!/usr/bin/env python3

"""Debug the _bound_wrapper issue."""

import jax.numpy as jnp
from flax import nnx
from functools import partial

class Model(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(3, 3, rngs=rngs)
        self.calls = jnp.array(0)

    def block(self, x, scale: float = 1.0):
        self.calls += 1
        return self.linear(x) * scale

def debug_bound_wrapper():
    """Debug what happens in the bound wrapper."""
    print("=== Debugging bound wrapper ===")
    
    m = Model(rngs=nnx.Rngs(0))
    x = jnp.ones((2, 3))
    
    # Test the partial creation
    f = partial(m.block, scale=2.0)
    print(f"Original partial: {f}")
    
    # Test our helper
    from flax.nnx.transforms.transforms import _maybe_unbind_and_rewrap_partial
    unbound_fn, bound_self = _maybe_unbind_and_rewrap_partial(f)
    
    print(f"Unbound function: {unbound_fn}")
    print(f"Bound self: {bound_self is not None}")
    
    # Now let's simulate what happens in the bound wrapper
    print("\n=== Simulating bound wrapper call ===")
    print(f"Call would be: inner_transform({bound_self}, {x.shape})")
    
    # The bound wrapper does: inner_transform(bound_self, *args, **kwargs)
    # Where args=(x,) and kwargs={}
    # So the call becomes: inner_transform(bound_self, x)
    # But unbound_fn is partial(Model.block, scale=2.0)
    # So merge_inputs gets called with partial(Model.block, scale=2.0)
    # Which means the function signature expects (self, x) since scale is bound
    # And we're calling it with (bound_self, x) which should work!
    
    # Let's test this step by step:
    print("\n=== Testing step by step ===")
    
    # Test calling the unbound function directly
    try:
        result = unbound_fn(bound_self, x)
        print(f"✓ Direct call to unbound_fn worked: {result.shape}")
    except Exception as e:
        print(f"✗ Direct call to unbound_fn failed: {e}")
    
    # Test with merge_inputs
    from flax.nnx.transforms import general
    try:
        merge_fn = general.merge_inputs(unbound_fn, ctxtag='test')
        print(f"✓ merge_inputs created successfully")
        
        # Now test calling it
        result = merge_fn(bound_self, x)
        print(f"✓ merge_inputs call worked")
    except Exception as e:
        print(f"✗ merge_inputs call failed: {e}")

if __name__ == "__main__":
    debug_bound_wrapper()