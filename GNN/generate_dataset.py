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

        # Multi-level voltage alarm flags
        # 0=normal, 1=mild overvoltage, 2=mild undervoltage, 3=severe undervoltage/overvoltage
        voltage_alarm = np.zeros_like(vm, dtype=int)
        # severe undervoltage/overvoltage
        voltage_alarm[(vm < 0.90) | (vm > 1.10)] = 3
        # mild undervoltage
        voltage_alarm[(vm >= 0.90) & (vm < 0.95)] = 2
        # mild overvoltage
        voltage_alarm[(vm > 1.05) & (vm <= 1.10)] = 1
        # normal (0) otherwise

        # Ensure each scenario has at least one voltage alarm (force one if none)
        if not np.any(voltage_alarm > 0):
            # pick a random bus and force a mild undervoltage alarm
            idx = rng.integers(0, len(voltage_alarm))
            voltage_alarm[idx] = 2
            # Optionally, adjust voltage to reflect the alarm (for realism)
            # Lower voltage slightly if not already
            if vm[idx] >= 0.95:
                vm[idx] = rng.uniform(0.91, 0.94)

        # Calculate line loading percent and thermal_class using varied limits
        loading_percent = n.res_line.loading_percent.values if "loading_percent" in n.res_line else np.full(len(n.line), np.nan)
        max_limits = n.line["max_loading_percent_varied"].values
        thermal_class = np.zeros_like(loading_percent, dtype=int)
        loading_percent_scaled = loading_percent * 100
        thermal_class[(loading_percent_scaled >= 95) & (loading_percent_scaled <= max_limits)] = 1
        thermal_class[loading_percent_scaled > max_limits] = 2

        # Flag buses connected to overloaded lines (thermal_class=2)
        overloaded_lines = np.where(thermal_class == 2)[0]
        overloaded_buses = set()
        for line_idx in overloaded_lines:
            overloaded_buses.add(n.line.from_bus.iloc[line_idx])
            overloaded_buses.add(n.line.to_bus.iloc[line_idx])
        thermal_alarm = np.array([2 if bus in overloaded_buses else 0 for bus in n.bus.index])

        # Combined bus alarm flag: max of voltage and thermal alarms per bus
        combined_alarm = np.maximum(voltage_alarm, thermal_alarm)

        all_buses.append(pd.DataFrame({
            "bus": n.bus.index.astype(int),
            "voltage": vm,
            "load_MW": p_load,
            "voltage_alarm": voltage_alarm,
            "thermal_alarm": thermal_alarm,
            "combined_alarm": combined_alarm,
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

    total_voltage_alarms = bus_df['voltage_alarm'].gt(0).sum()
    total_thermal_alarms = bus_df['thermal_alarm'].gt(0).sum()
    total_combined_alarms = bus_df['combined_alarm'].gt(0).sum()

    bus_df.to_csv("bus_scenarios.csv", index=False)
    edge_df.to_csv("edge_scenarios.csv", index=False)
    # Create and save unlabeled versions for prediction
    bus_inputs = bus_df.drop(columns=["voltage_alarm", "thermal_alarm", "combined_alarm"])
    bus_inputs.to_csv("bus_inputs.csv", index=False)
    edge_inputs = edge_df.drop(columns=["in_service", "thermal_class"]) if "in_service" in edge_df.columns else edge_df.copy()
    edge_inputs.to_csv("edge_inputs.csv", index=False)

    print(f"✅ Generated {len(bus_df)} bus rows = {n_buses} buses × {n_scenarios} scenarios.")
    print(f"   Edges: {len(edge_df)} rows (~{edges_mean:.1f} per scenario, min {edges_min}, max {edges_max}).")
    print(f"⚠️  Total voltage alarms: {total_voltage_alarms}")
    print(f"🔥 Total thermal alarms: {total_thermal_alarms}")
    print(f"🚨 Total combined alarms: {total_combined_alarms}")
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