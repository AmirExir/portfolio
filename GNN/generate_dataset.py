import copy
import pandapower.networks as pn
import pandas as pd
import numpy as np
import pandapower as pp
import argparse

def build_ieee118():
    net = pn.case118()   # built-in IEEE-118 test system
    return net

def sample_scenarios(net, n_scen=50, outage_p=0.03, load_sigma=0.1, seed=42, use_numba=False):
    rng = np.random.default_rng(seed)
    all_buses, all_edges = [], []

    for s in range(n_scen):
        n = copy.deepcopy(net)

        # Dynamically scale loads between 1.1×–1.4× to simulate stressed conditions
        if len(n.load):
            scale_factors = rng.uniform(1.1, 1.4, len(n.load))
            n.load["p_mw"] *= scale_factors

        # Randomly perform N-1, N-2, or N-3 outages per scenario
        if len(n.line):
            n_lines = len(n.line)
            k = rng.choice([1, 2, 3])  # N-1, N-2, N-3
            k = min(k, n_lines)  # in case system is small
            outage_idx = rng.choice(n_lines, size=k, replace=False)
            outage_mask = np.zeros(n_lines, dtype=bool)
            outage_mask[outage_idx] = True
            n.line.in_service = ~outage_mask

        # Add slight randomness to line thermal limits to simulate realistic variability
        if "max_loading_percent" in n.line.columns:
            variability = rng.normal(loc=0.0, scale=2.0, size=len(n.line))  # ±2% variability
            n.line["max_loading_percent_varied"] = n.line["max_loading_percent"] * (1 + variability / 100)
        else:
            n.line["max_loading_percent_varied"] = 100.0  # default if not present

        # run power flow
        try:
            pp.runpp(n, numba=use_numba)
        except Exception:
            continue  # skip infeasible scenarios

        vm = n.res_bus.vm_pu.values
        p_load = n.load.groupby("bus").p_mw.sum().reindex(n.bus.index, fill_value=0).values

        # Define bus voltage_class:
        # 0.95 ≤ V ≤ 1.05 → class 0
        # 0.90 ≤ V < 0.95 or 1.05 < V ≤ 1.10 → class 1
        # V < 0.90 or V > 1.10 → class 2
        voltage_class = np.zeros_like(vm, dtype=int)
        voltage_class[(vm < 0.90) | (vm > 1.10)] = 2
        voltage_class[((vm >= 0.90) & (vm < 0.95)) | ((vm > 1.05) & (vm <= 1.10))] = 1
        # class 0 otherwise

        # Calculate line loading percent and thermal_class using varied limits
        loading_percent = n.res_line.loading_percent.values if "loading_percent" in n.res_line else np.full(len(n.line), np.nan)

        loading_percent *= 100.0  # Force conversion to percentage for consistency

        max_limits = n.line["max_loading_percent_varied"].values
        # Define line thermal_class:
        # ≤ 90 → class 0
        # 90 < loading ≤ 100 → class 1
        # > 100 → class 2
        thermal_class = np.zeros_like(loading_percent, dtype=int)
        thermal_class[loading_percent > 90] = 1
        thermal_class[loading_percent > 100] = 2

        all_buses.append(pd.DataFrame({
            "bus": n.bus.index.astype(int),
            "voltage": vm,
            "load_MW": p_load,
            "voltage_class": voltage_class,
            "scenario": s
        }))

        all_edges.append(pd.DataFrame({
            "from_bus": n.line.from_bus.values,
            "to_bus": n.line.to_bus.values,
            "x_pu": n.line.x_ohm_per_km.values if "x_ohm_per_km" in n.line else np.nan,
            "in_service": n.line.in_service.values,
            "length_km": n.line.length_km.values if "length_km" in n.line else np.nan,
            "loading_percent": loading_percent,
            "thermal_class": thermal_class,
            "scenario": s
        }))

    bus_df = pd.concat(all_buses, ignore_index=True)
    edge_df = pd.concat(all_edges, ignore_index=True)

    n_scenarios = bus_df['scenario'].nunique()
    buses_per_s = bus_df.groupby('scenario')['bus'].nunique()
    edges_per_s = edge_df.groupby('scenario')[['from_bus','to_bus']].size()
    n_buses = int(buses_per_s.iloc[0]) if not buses_per_s.empty else 0
    edges_mean = float(edges_per_s.mean()) if len(edges_per_s) > 0 else 0.0
    edges_min = int(edges_per_s.min()) if len(edges_per_s) > 0 else 0
    edges_max = int(edges_per_s.max()) if len(edges_per_s) > 0 else 0

    total_voltage_alarms = bus_df['voltage_class'].gt(0).sum()
    total_thermal_alarms = edge_df['thermal_class'].gt(0).sum()

    bus_df.to_csv("bus_scenarios.csv", index=False)
    edge_df.to_csv("edge_scenarios.csv", index=False)
    # Create and save unlabeled versions for prediction
    bus_inputs = bus_df.drop(columns=["voltage", "voltage_class"])
    bus_inputs.to_csv("bus_inputs.csv", index=False)
    if "in_service" in edge_df.columns:
        edge_inputs = edge_df.drop(columns=["in_service", "loading_percent", "thermal_class"])
    else:
        edge_inputs = edge_df.drop(columns=["loading_percent", "thermal_class"])
    edge_inputs.to_csv("edge_inputs.csv", index=False)

    print(f"✅ Generated {len(bus_df)} bus rows = {n_buses} buses × {n_scenarios} scenarios.")
    print(f"   Edges: {len(edge_df)} rows (~{edges_mean:.1f} per scenario, min {edges_min}, max {edges_max}).")
    print(f"⚠️  Total bus voltage alarms (class>0): {total_voltage_alarms}")
    print(f"🔥 Total line thermal alarms (class>0): {total_thermal_alarms}")
    print("🟢 Saved labeled datasets: bus_scenarios.csv and edge_scenarios.csv")
    print("🟢 Saved unlabeled prediction-ready datasets: bus_inputs.csv and edge_inputs.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=200)
    parser.add_argument("--outage-p", type=float, default=0.03)
    parser.add_argument("--load-sigma", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--numba", action='store_true', default=False)
    args = parser.parse_args()

    net = build_ieee118()
    sample_scenarios(net, n_scen=args.scenarios, outage_p=args.outage_p, load_sigma=args.load_sigma, seed=args.seed, use_numba=args.numba)