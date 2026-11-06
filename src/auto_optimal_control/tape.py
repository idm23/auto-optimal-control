"""Large expressions as a tape: a straight-line program with shared steps.

A robot's torques, written in SymPy, are small as a graph and enormous as a
tree: the recursive Newton-Euler sum of a six-joint arm is about a thousand
operations and a hundred thousand when written out, and SymPy's ``diff`` and
``simplify`` work on the written-out form. A :class:`Tape` keeps the graph.
Every step is a value computed once from earlier steps, equal steps are merged
when they are recorded, and a derivative is more steps, so differentiating
costs the size of the graph, not of the tree.
"""

import math

import sympy as sp

CHUNK = 500              # statements per generated helper function
OUTLINED_PREFIX = "o_"   # the C local holding an outlined step's value
_UNARY = {sp.sin: "sin", sp.cos: "cos", sp.tan: "tan", sp.exp: "exp", sp.log: "log"}


class Unsupported(ValueError):
    """The expression uses an operation a tape does not record."""


class Tape:
    """Steps of a straight-line program over named inputs.

    A step is an integer id. Ids only refer to smaller ids, so the order of
    the ids is an order to evaluate them in.
    """

    def __init__(self):
        self.steps = []          # (op, *args): ("input", name), ("const", value), ...
        self._ids = {}           # step -> id, so equal steps are merged
        self._depends = []       # id -> bitmask of the inputs it depends on
        self.inputs = {}         # name -> id
        self._bit = {}           # input id -> bit
        self._sympy = {}         # SymPy node -> id
        self._grad = {}          # input id -> {step id -> id of its derivative}
        self.symbols = {}        # input name -> SymPy symbol
        self._outlined_class = None
        self.zero = self.const(0.0)
        self.one = self.const(1.0)

    def __len__(self):
        return len(self.steps)

    def _record(self, step, depends):
        i = self._ids.get(step)
        if i is None:
            i = len(self.steps)
            self.steps.append(step)
            self._ids[step] = i
            self._depends.append(depends)
        return i

    def const(self, value):
        return self._record(("const", float(value)), 0)

    def input(self, name):
        i = self.inputs.get(name)
        if i is None:
            i = self._record(("input", name), 1 << len(self.inputs))
            self.inputs[name] = i
            self._bit[i] = 1 << (len(self.inputs) - 1)
        return i

    def value(self, i):
        """The value of step ``i`` if it is a constant, else None."""
        step = self.steps[i]
        return step[1] if step[0] == "const" else None

    def depends_on(self, i, name):
        """Whether step ``i`` depends on input ``name`` (by structure)."""
        return name in self.inputs and bool(self._depends[i] & self._bit[self.inputs[name]])

    def add(self, a, b):
        x, y = self.value(a), self.value(b)
        if x is not None and y is not None:
            return self.const(x + y)
        if x == 0.0:
            return b
        if y == 0.0:
            return a
        an, bn = self.steps[a][0] == "neg", self.steps[b][0] == "neg"
        if an and bn:
            return self.neg(self.add(self.steps[a][1], self.steps[b][1]))
        if bn:
            return self.sub(a, self.steps[b][1])
        if an:
            return self.sub(b, self.steps[a][1])
        if a > b:
            a, b = b, a
        return self._record(("add", a, b), self._depends[a] | self._depends[b])

    def neg(self, a):
        x = self.value(a)
        if x is not None:
            return self.const(-x)
        if self.steps[a][0] == "neg":
            return self.steps[a][1]
        return self._record(("neg", a), self._depends[a])

    def sub(self, a, b):
        if a == b:
            return self.zero
        x, y = self.value(a), self.value(b)
        if x is not None and y is not None:
            return self.const(x - y)
        if y == 0.0:
            return a
        if x == 0.0:
            return self.neg(b)
        if self.steps[b][0] == "neg":
            return self.add(a, self.steps[b][1])
        if self.steps[a][0] == "neg":
            return self.neg(self.add(self.steps[a][1], b))
        if a > b:              # a - b and b - a are one step, negated
            return self.neg(self.sub(b, a))
        return self._record(("sub", a, b), self._depends[a] | self._depends[b])

    def mul(self, a, b):
        x, y = self.value(a), self.value(b)
        if x is not None and y is not None:
            return self.const(x * y)
        if x == 0.0 or y == 0.0:
            return self.zero
        if x == 1.0:
            return b
        if y == 1.0:
            return a
        if x == -1.0:
            return self.neg(b)
        if y == -1.0:
            return self.neg(a)
        if self.steps[a][0] == "neg":
            return self.neg(self.mul(self.steps[a][1], b))
        if self.steps[b][0] == "neg":
            return self.neg(self.mul(a, self.steps[b][1]))
        if x is not None and x < 0.0:
            return self.neg(self.mul(self.const(-x), b))
        if y is not None and y < 0.0:
            return self.neg(self.mul(a, self.const(-y)))
        if a > b:
            a, b = b, a
        return self._record(("mul", a, b), self._depends[a] | self._depends[b])

    def div(self, a, b):
        x, y = self.value(a), self.value(b)
        if x == 0.0:
            return self.zero
        if y is not None:
            return self.mul(a, self.const(1.0 / y))
        return self._record(("div", a, b), self._depends[a] | self._depends[b])

    def unary(self, op, a):
        x = self.value(a)
        if x is not None:
            return self.const(float(getattr(sp, op)(x)))
        return self._record((op, a), self._depends[a])

    def power(self, a, k):
        """``a ** k`` for an integer ``k``, as multiplications."""
        if k < 0:
            return self.div(self.one, self.power(a, -k))
        out, base = self.one, a
        while k:
            if k & 1:
                out = self.mul(out, base)
            base = self.mul(base, base)
            k >>= 1
        return out

    def from_sympy(self, expr):
        """Record a SymPy expression; each distinct subexpression is recorded once."""
        stack = [sp.sympify(expr)]
        memo = self._sympy
        while stack:
            e = stack[-1]
            if e in memo:
                stack.pop()
                continue
            if e.is_Number:
                memo[e] = self.const(float(e))
            elif isinstance(e, sp.Symbol):
                memo[e] = self.input(e.name)
                self.symbols[e.name] = e
            elif e.is_NumberSymbol:
                memo[e] = self.const(float(e))
            else:
                if not (e.is_Add or e.is_Mul or e.is_Pow or type(e) in _UNARY):
                    raise Unsupported(f"{type(e).__name__} cannot be put on a tape")
                missing = [a for a in e.args if a not in memo]
                if missing:
                    stack.extend(missing)
                    continue
                memo[e] = self._build(e)
            stack.pop()
        return memo[sp.sympify(expr)]

    def _build(self, e):
        memo = self._sympy
        if e.is_Add:
            out = self.zero
            for a in e.args:
                out = self.add(out, memo[a])
            return out
        if e.is_Mul:
            out = self.one
            for a in e.args:
                out = self.mul(out, memo[a])
            return out
        if e.is_Pow:
            base, power = e.args
            if power.is_Integer:
                return self.power(memo[base], int(power))
            return self._record(("pow", memo[base], memo[power]),
                                self._depends[memo[base]] | self._depends[memo[power]])
        return self.unary(_UNARY[type(e)], memo[e.args[0]])

    def diff(self, i, name):
        """The step of ``d(step i)/d(input name)``, recorded on this tape."""
        if not self.depends_on(i, name):
            return self.zero
        var = self.inputs[name]
        grad = self._grad.setdefault(var, {})
        if i in grad:
            return grad[i]
        bit = self._bit[var]
        stack = [i]
        while stack:
            k = stack[-1]
            if k in grad:
                stack.pop()
                continue
            args = [a for a in self.steps[k][1:] if isinstance(a, int)]
            if self.steps[k][0] == "input" or self.steps[k][0] == "const":
                args = []
            missing = [a for a in args if self._depends[a] & bit and a not in grad]
            if missing:
                stack.extend(missing)
                continue
            d = {a: grad[a] if self._depends[a] & bit else self.zero for a in args}
            grad[k] = self._derivative(k, var, d)
            stack.pop()
        return grad[i]

    def _derivative(self, k, var, d):
        step = self.steps[k]
        op = step[0]
        if op == "input":
            return self.one if k == var else self.zero
        a = step[1]
        if op == "add":
            return self.add(d[a], d[step[2]])
        if op == "sub":
            return self.sub(d[a], d[step[2]])
        if op == "neg":
            return self.neg(d[a])
        if op == "mul":
            b = step[2]
            return self.add(self.mul(d[a], b), self.mul(a, d[b]))
        if op == "div":
            b = step[2]
            return self.div(self.sub(d[a], self.mul(k, d[b])), b)
        if op == "sin":
            return self.mul(self.unary("cos", a), d[a])
        if op == "cos":
            return self.neg(self.mul(self.unary("sin", a), d[a]))
        if op == "tan":
            return self.mul(self.add(self.one, self.mul(k, k)), d[a])
        if op == "exp":
            return self.mul(k, d[a])
        if op == "log":
            return self.div(d[a], a)
        if op == "pow":
            b = step[2]
            power = self.mul(self.div(self.mul(b, d[a]), a), k)
            if d[b] == self.zero:
                return power
            return self.add(power, self.mul(self.mul(self.unary("log", a), d[b]), k))
        raise Unsupported(op)

    def directional(self, i, seeds):
        """``sum d(step i)/d(input) * seed``: the derivative along ``seeds``, a dict
        from input name to step, in one forward pass."""
        bits = 0
        for name in seeds:
            if name in self.inputs:
                bits |= self._bit[self.inputs[name]]
        if not self._depends[i] & bits:
            return self.zero
        seen = {}
        stack = [i]
        while stack:
            k = stack[-1]
            if k in seen:
                stack.pop()
                continue
            step = self.steps[k]
            if step[0] == "input":
                seen[k] = seeds.get(step[1], self.zero)
                stack.pop()
                continue
            args = [a for a in step[1:]]
            missing = [a for a in args if self._depends[a] & bits and a not in seen]
            if missing:
                stack.extend(missing)
                continue
            d = {a: seen[a] if self._depends[a] & bits else self.zero for a in args}
            seen[k] = self._derivative(k, -1, d)
            stack.pop()
        return seen[i]

    def gradient(self, i, names):
        """``[d(step i)/d(name) for name in names]`` in one reverse pass."""
        adjoint = {i: self.one}
        for k in range(i, -1, -1):
            a = adjoint.get(k)
            if a is None or a == self.zero:
                continue
            step = self.steps[k]
            op = step[0]
            if op in ("input", "const"):
                continue
            pushes = self._pullback(k, a)
            for arg, value in pushes:
                if self._depends[arg]:
                    adjoint[arg] = self.add(adjoint.get(arg, self.zero), value)
        return [adjoint.get(self.inputs[n], self.zero) if n in self.inputs else self.zero
                for n in names]

    def _pullback(self, k, a):
        step = self.steps[k]
        op, x = step[0], step[1]
        if op == "add":
            return [(x, a), (step[2], a)]
        if op == "sub":
            return [(x, a), (step[2], self.neg(a))]
        if op == "neg":
            return [(x, self.neg(a))]
        if op == "mul":
            return [(x, self.mul(a, step[2])), (step[2], self.mul(a, x))]
        if op == "div":
            y = step[2]
            q = self.div(a, y)
            return [(x, q), (y, self.neg(self.mul(q, k)))]
        if op == "sin":
            return [(x, self.mul(a, self.unary("cos", x)))]
        if op == "cos":
            return [(x, self.neg(self.mul(a, self.unary("sin", x))))]
        if op == "tan":
            return [(x, self.mul(a, self.add(self.one, self.mul(k, k))))]
        if op == "exp":
            return [(x, self.mul(a, k))]
        if op == "log":
            return [(x, self.div(a, x))]
        if op == "pow":
            y = step[2]
            pushes = [(x, self.mul(a, self.div(self.mul(y, k), x)))]
            if self._depends[y]:
                pushes.append((y, self.mul(a, self.mul(self.unary("log", x), k))))
            return pushes
        raise Unsupported(op)

    def reachable(self, outputs):
        """The non-constant, non-input steps ``outputs`` need, in evaluation order."""
        seen, stack = set(), [i for i in outputs]
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            if self.steps[k][0] not in ("const", "input"):
                stack.extend(self.steps[k][1:])
        return sorted(k for k in seen if self.steps[k][0] not in ("const", "input"))

    def operand(self, i, names):
        """The C text of step ``i`` as an operand: a constant, an input, or its local name."""
        step = self.steps[i]
        if step[0] == "const":
            return repr(step[1]) if step[1] >= 0 else f"({step[1]!r})"
        if step[0] == "input":
            return names[step[1]]
        return f"s{i}"

    def _statement(self, k, operand):
        step = self.steps[k]
        op = step[0]
        infix = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        if op in infix:
            return f"{operand(step[1])} {infix[op]} {operand(step[2])}"
        if op == "neg":
            return f"-{operand(step[1])}"
        if op == "pow":
            return f"pow({operand(step[1])}, {operand(step[2])})"
        return f"{op}({operand(step[1])})"

    def c_block(self, outputs, names, prefix, args="t, z, p, u, mu", chunk=CHUNK):
        """C for ``outputs``: ``(helpers, lines, operand)``.

        ``lines`` go in the calling function and define what ``operand(step)``
        names. A long program is cut into ``helpers``: a compiler's time grows
        faster than the length of one function, and 60000 statements take
        minutes as one and seconds in pieces. ``names`` maps input names to
        C text over ``args``; ``prefix`` makes the helpers' names unique.
        """
        order = self.reachable(outputs)
        wanted = set(outputs)
        if len(order) <= chunk:
            local = lambda i: self.operand(i, names)
            lines = [f"    const double s{k} = {self._statement(k, local)};" for k in order]
            return [], lines, local

        pieces = [order[i:i + chunk] for i in range(0, len(order), chunk)]
        piece_of = {k: n for n, piece in enumerate(pieces) for k in piece}
        # A value crosses pieces through an array; within a piece it is a local.
        shared = {k for k in order if k in wanted}
        for k in order:
            for a in self.steps[k][1:]:
                if a in piece_of and piece_of[a] != piece_of[k]:
                    shared.add(a)
        slot = {k: n for n, k in enumerate(sorted(shared))}

        def within(i):
            return f"v_[{slot[i]}]" if i in slot else self.operand(i, names)

        helpers = []
        signature = ("double t, const double* z, const double* p, const double* u, "
                     "const double* mu, double* restrict v_")
        lines = [f"    double v_[{max(len(slot), 1)}];", "    (void)v_;"]
        for n, piece in enumerate(pieces):
            body = [f"static void {prefix}_{n}({signature}) {{",
                    "    (void)t; (void)z; (void)p; (void)u; (void)mu;"]
            for k in piece:
                text = self._statement(k, within)
                body.append(f"    v_[{slot[k]}] = {text};" if k in slot
                            else f"    const double s{k} = {text};")
            body.append("}")
            helpers.append("\n".join(body))
            lines.append(f"    {prefix}_{n}({args}, v_);")
        return helpers, lines, within

    def evaluate(self, outputs, values):
        """The values of ``outputs``, with ``values`` mapping input names to numbers."""
        fns = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp,
               "log": math.log}
        v = {}

        def get(i):
            step = self.steps[i]
            if step[0] == "const":
                return step[1]
            if step[0] == "input":
                return values[step[1]]
            return v[i]

        for k in self.reachable(outputs):
            step = self.steps[k]
            op = step[0]
            if op == "add":
                v[k] = get(step[1]) + get(step[2])
            elif op == "sub":
                v[k] = get(step[1]) - get(step[2])
            elif op == "mul":
                v[k] = get(step[1]) * get(step[2])
            elif op == "div":
                v[k] = get(step[1]) / get(step[2])
            elif op == "neg":
                v[k] = -get(step[1])
            elif op == "pow":
                v[k] = get(step[1]) ** get(step[2])
            else:
                v[k] = fns[op](get(step[1]))
        return [get(i) for i in outputs]

    def input_names(self, i):
        """The inputs step ``i`` depends on, in the order they were first seen."""
        return [n for n, k in self.inputs.items() if self._depends[i] & self._bit[k]]

    def outlined(self, i, symbols=None):
        """A SymPy expression for step ``i``, written around it instead of out of it.

        ``symbols`` maps input names to what stands for them (default: the
        symbols they were recorded from).
        """
        symbols = self.symbols if symbols is None else symbols
        step = self.steps[i]
        if step[0] == "const":
            return sp.nsimplify(step[1], rational=True) if step[1] in (0.0, 1.0, -1.0) \
                else sp.Float(step[1])
        if step[0] == "input":
            return symbols[step[1]]
        if step[0] == "neg":
            return -self.outlined(step[1], symbols)
        cls = self._outlined_class
        if cls is None:
            cls = self._outlined_class = type("Outlined", (Outlined,), {"tape": self})
        return cls(sp.Integer(i), *[symbols[n] for n in self.input_names(i)])


class Outlined(sp.Function):
    """A step of a :class:`Tape` standing in a SymPy expression as one opaque function.

    Its arguments are the step's id and the inputs it depends on, in the order
    of :meth:`Tape.input_names`, so substituting for a symbol still reaches it
    and differentiating gives another step, or zero where the step does not
    depend on the variable. It prints as the C name of the step's value.
    """

    tape = None

    def fdiff(self, argindex=1):
        step = int(self.args[0])
        names = self.tape.input_names(step)
        # Whatever has been substituted for the inputs carries over to the
        # derivative, so a second derivative after a substitution is not zero.
        return self.tape.outlined(self.tape.diff(step, names[argindex - 2]),
                                  dict(zip(names, self.args[1:])))

    def _eval_is_extended_real(self):
        return True

    def _ccode(self, printer):
        return f"{OUTLINED_PREFIX}{int(self.args[0])}"
