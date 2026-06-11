# Circuit Language

One statement per line; `#` starts a comment. Qubits are `q0, q1, ...`,
classical bits `c0, c1, ...`.

```
circuit bell:          # optional name header
qubits 2               # required before any instruction
bits 2                 # optional; inferred from measurements otherwise
h q0
cx q0 q1
measure all
```

## Statements

| statement | meaning |
|-----------|---------|
| `circuit <name>:` | names the circuit (optional) |
| `qubits <n>` | declares the register size (required) |
| `bits <n>` | declares classical bits (optional) |
| `<gate> q<i> [q<j>]` | applies a gate |
| `measure all` | measures every qubit into the same-index clbit |
| `measure q<i>` | measures one qubit into clbit *i* |
| `measure q<i> -> c<j>` | measures into an explicit clbit |

## Gates

- Single-qubit: `i x y z h s t`
- Rotations: `rx ry rz` — either `rx(pi/2) q0` or `rx 1.5708 q0`.
  Angles accept `pi` and simple arithmetic (`pi/2`, `2*pi`).
- Two-qubit: `cx` (alias `cnot`), `cz`, `swap`

## Errors

The parser raises `DSLParseError` with the offending line number, e.g.
`line 3: unknown gate 'frobnicate'`.

## IR JSON

`POST /circuits/parse` (or `CircuitIR.to_json()`) yields:

```json
{
  "name": "bell",
  "num_qubits": 2,
  "num_clbits": 2,
  "instructions": [
    {"name": "h", "qubits": [0], "clbits": [], "params": []},
    {"name": "cx", "qubits": [0, 1], "clbits": [], "params": []},
    {"name": "measure", "qubits": [0], "clbits": [0], "params": []},
    {"name": "measure", "qubits": [1], "clbits": [1], "params": []}
  ]
}
```

Bitstring convention everywhere in the system: **qubit 0 is the leftmost
character** (`"10"` means q0=1, q1=0).
