#!/usr/bin/env python3

"""Test script to reproduce the remat issue with bound methods."""

import jax
import jax.numpy as jnp
from flax import nnx

class Model(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(16, 32, rngs=rngs)
        self.x_max = jnp.array(0.0)

    def __call__(self, x):
        # This should fail - bound method
        return nnx.remat(self.block)(x)

    def call_correct(self, x):
        # This should work - unbound method
        return nnx.remat(self.block.__func__)(self, x)

    def block(self, x):
        self.x_max = jnp.maximum(self.x_max, x.max())
        return self.linear(x)


def test_bound_method_issue():
    """Test the bound method issue."""
    print("Testing bound method issue...")
    
    model = Model(rngs=nnx.Rngs(0))
    x = jnp.ones((3, 16))
    
    try:
        result = model(x)
        print(f"UNEXPECTED: Bound method worked! Result shape: {result.shape}")
        return False
    except Exception as e:
        print(f"EXPECTED: Bound method failed with: {type(e).__name__}: {e}")
        
    try:
        result = model.call_correct(x)
        print(f"EXPECTED: Unbound method worked! Result shape: {result.shape}")
        return True
    except Exception as e:
        print(f"UNEXPECTED: Unbound method failed with: {type(e).__name__}: {e}")
        return False


def test_method_inspection():
    """Understand the difference between bound and unbound methods."""
    print("\nInspecting method types...")
    
    model = Model(rngs=nnx.Rngs(0))
    
    bound_method = model.block
    unbound_method = model.block.__func__
    
    print(f"Bound method: {bound_method}")
    print(f"Bound method type: {type(bound_method)}")
    print(f"Bound method __self__: {getattr(bound_method, '__self__', 'No __self__')}")
    print(f"Bound method __func__: {getattr(bound_method, '__func__', 'No __func__')}")
    
    print(f"\nUnbound method: {unbound_method}")
    print(f"Unbound method type: {type(unbound_method)}")
    print(f"Unbound method __self__: {getattr(unbound_method, '__self__', 'No __self__')}")


def test_remat_signature():
    """Test what remat expects."""
    print("\nTesting remat function signature expectations...")
    
    def simple_func(x):
        return x * 2
    
    def bound_func_simulation(x):
        # Simulate what happens when we pass a bound method
        return x * 2
    
    try:
        # Test with a simple function
        remat_simple = nnx.remat(simple_func)
        result = remat_simple(jnp.array(5.0))
        print(f"Simple function with remat worked: {result}")
    except Exception as e:
        print(f"Simple function with remat failed: {e}")


if __name__ == "__main__":
    test_bound_method_issue()
    test_method_inspection()
    test_remat_signature()