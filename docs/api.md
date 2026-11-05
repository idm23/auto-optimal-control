# API reference

## Posing a problem

```{eval-rst}
.. automodule:: auto_optimal_control.problem
```

## The necessary conditions

```{eval-rst}
.. automodule:: auto_optimal_control.pontryagin
```

## Compiled kernels

```{eval-rst}
.. automodule:: auto_optimal_control.build
```

The functions of the compiled module itself (`propagate`, `trajectory`,
`propagate_batch`, `jacobian`, `switching`, `stm_jump` and the rest) are
reached as attributes of {class}`~auto_optimal_control.build.Kernels`, and
each carries its own docstring: use `help(kernels.propagate)`.

## Shooting

```{eval-rst}
.. automodule:: auto_optimal_control.shooting
```

## Solving from a costate distribution

```{eval-rst}
.. automodule:: auto_optimal_control.homotopy
```

## NumPy evaluation

```{eval-rst}
.. automodule:: auto_optimal_control.numeric
```

## Code generation

```{eval-rst}
.. automodule:: auto_optimal_control.codegen
```

## Examples

```{eval-rst}
.. automodule:: auto_optimal_control.examples
```
