# pnnl-dopf-admm

![Version](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/openEDI/pnnl-dopf-admm/main/pyproject.toml&query=$.project.version&label=version&color=blue)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OEDISI 3.0](https://img.shields.io/badge/oedisi-~3.0-green.svg)](https://github.com/openEDI/oedisi)

OEDISI Distributed Optimal Power Flow (DOPF) Federate using spatial decomposition and the Exact-Network-Approximation-based Parallel Power Flow (ENAPP) ADMM algorithm powered by [`distopf`](https://github.com/GRIDAPPSD/distopf).

---

## Overview

Each instance of this federate represents **one spatial decomposition area** of a distribution network. On each HELICS iteration cycle, the federate:
1. Receives boundary condition updates ($V_{dn}$ swing voltages from parent areas and $S_{up}$ power injections from child areas) from hub aggregators.
2. Updates local sub-network loads and generation from feeder measurements.
3. Solves a local OPF problem using `distopf` and CVXPY/Clarabel according to the area's configured **objective**.
4. Publishes boundary variables ($S_{up}$ and $V_{dn}$) to hub aggregators and control change commands to the controlled feeder.

```
                  ┌──────────────────────┐
                  │    pnnl-hub-power    │
                  │   pnnl-hub-voltage   │
                  │   pnnl-hub-control   │
                  └──────────┬───────────┘
                             │ (S_up, V_dn, C)
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Area 0 (Root) │    │    Area 1     │    │    Area 2     │
│ minimize_loss │    │minimize_curtail│   │ maximize_gen  │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## Area Objective Configuration

Each area federate can be configured with an independent optimization objective via the `objective` static input parameter.

| Objective | Description | Formulation |
| :--- | :--- | :--- |
| `maximize_gen` *(Default)* | Maximizes active power generation from distributed energy resources (DERs / PV systems). | $\min \sum -P_{\text{gen}}$ |
| `minimize_loss` | Minimizes total branch $I^2R$ resistive power losses across the area's lines. | $\min \sum r_{ij} (P_{ij}^2 + Q_{ij}^2)$ |
| `minimize_curtail` | Minimizes quadratic curtailment of DER active power from available generation capacity. | $\min \sum (P_{\text{max}} - P_{\text{gen}})^2$ |
| `minimize_load` | Minimizes net active power imported at the area's substation / swing bus. | $\min \sum P_{\text{swing}}$ |

> [!NOTE]
> Passive / unoptimized power flow (`none`) is disallowed. Unrecognized objective strings will raise a Pydantic `ValidationError` at configuration time.

---

## Configuration Parameters

### Static Inputs (`static_inputs.json`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` | *Required* | Unique identifier for the federate instance (e.g. `pnnl_dopf_admm_0`). |
| `objective` | `str` | `"maximize_gen"` | OPF objective: `"maximize_gen"`, `"minimize_loss"`, `"minimize_curtail"`, or `"minimize_load"`. |
| `source_bus` | `str` | `""` | Boundary source bus name (e.g. `"150"`) or boundary switch ID (e.g. `"sw3"`). |
| `source_line` | `str` | `""` | Boundary line or switch ID connected to the source bus. |
| `switches` | `list[str]` | `[]` | List of controllable switch IDs bounding this area. |
| `control_type` | `str` | `"real"` | Control mode (`"real"` or `"reactive"`). |
| `vup_tol` | `float` | `0.001` | ADMM convergence tolerance for upstream voltage mismatch. |
| `sdn_tol` | `float` | `0.001` | ADMM convergence tolerance for power mismatch. |
| `rho_vup` | `float` | `1000.0` | ADMM penalty parameter for upstream voltage. |
| `rho_sup` | `float` | `0.0` | ADMM penalty parameter for active power injection. |
| `rho_vdn` | `float` | `0.0` | ADMM penalty parameter for downstream voltage discrepancy. |
| `rho_sdn` | `float` | `1000.0` | ADMM penalty parameter for power flow discrepancy. |
| `max_itr` | `int` | `100` | Maximum ADMM iterations per timestep. |
| `relaxed` | `bool` | `false` | Enable relaxed ADMM formulation. |
| `number_of_timesteps` | `int` | `1` | Total simulation timesteps. |
| `deltat` | `float` | `3600.0` | Co-simulation time step interval in seconds. |

### Dynamic Inputs (Subscriptions)

| Port ID | Data Type | Source |
| :--- | :--- | :--- |
| `topology` | `Topology` | Feeder topology message. |
| `injections` | `Injection` | Live nodal load and generation measurements. |
| `voltages_real` | `VoltagesReal` | Real voltage measurements. |
| `voltages_imag` | `VoltagesImaginary` | Imaginary voltage measurements. |
| `sub_v` | `VoltagesMagnitude` | Aggregated boundary swing voltages from `pnnl-hub-voltage`. |
| `sub_p` | `PowersReal` | Aggregated boundary active power injections from `pnnl-hub-power`. |
| `sub_q` | `PowersImaginary` | Aggregated boundary reactive power injections from `pnnl-hub-power`. |

### Dynamic Outputs (Publications)

| Port ID | Data Type | Description |
| :--- | :--- | :--- |
| `pub_c` | `CommandList` | Inverter setpoint change commands for the control feeder. |
| `pub_v` | `VoltagesMagnitude` | Downstream boundary voltages ($V_{dn}$) published to `pnnl-hub-voltage`. |
| `pub_p` | `PowersReal` | Upstream boundary active power ($P_{up}$) published to `pnnl-hub-power`. |
| `pub_q` | `PowersImaginary` | Upstream boundary reactive power ($Q_{up}$) published to `pnnl-hub-power`. |
| `controls_real` | `PowersReal` | Active power generator control setpoints. |
| `controls_imag` | `PowersImaginary` | Reactive power generator control setpoints. |
| `voltages_mag` | `VoltagesMagnitude` | Area voltage magnitudes (per-unit). |
| `powers_mag` | `PowersMagnitude` | Branch power flow magnitudes (kVA). |
| `powers_ang` | `PowersAngle` | Branch power flow angles (degrees). |
| `solver_stats` | `MeasurementArray` | Solver convergence metrics and ADMM iteration stats. |

---

## Scenarios

### 1. Multi-Objective IEEE 123 Bus Scenario
File: [`scenarios/pnnl_dopf_admm_ieee123_multi_objective.json`](file:///home/tslay/dev/oedisi-components/Components/pnnl-dopf-admm/scenarios/pnnl_dopf_admm_ieee123_multi_objective.json)

Demonstrates heterogeneous objective coordination across 5 spatial areas:
* **Area 0** (`pnnl_dopf_admm_0`, source bus 150): `"minimize_loss"`
* **Area 1** (`pnnl_dopf_admm_1`, source bus 18): `"minimize_curtail"`
* **Area 2** (`pnnl_dopf_admm_2`, source bus 13): `"maximize_gen"`
* **Area 3** (`pnnl_dopf_admm_3`, source bus 60): `"minimize_load"`
* **Area 4** (`pnnl_dopf_admm_4`, source bus 97): `"maximize_gen"`

### 2. Standard IEEE 123 Bus Scenario
File: [`scenarios/pnnl_dopf_admm_ieee123.json`](file:///home/tslay/dev/oedisi-components/Components/pnnl-dopf-admm/scenarios/pnnl_dopf_admm_ieee123.json)

Standard baseline scenario using default `maximize_gen` for all areas.

---

## Running & Testing

### Unit Tests
Run the component test suite with `pytest`:

```bash
uv run pytest tests/ -v
```

### Building and Running Co-Simulations
From the `oedisi-components` root:

```bash
# Build the multi-objective federation
uv run oedisi build --system Components/pnnl-dopf-admm/scenarios/pnnl_dopf_admm_ieee123_multi_objective.json

# Run the co-simulation
uv run oedisi run
```
