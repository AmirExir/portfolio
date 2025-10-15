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

        # jitter loads ±10%
        if len(n.load):
            n.load["p_mw"] *= (1.0 + rng.normal(0, load_sigma, len(n.load)))

        # randomly open ~3% of lines
        if len(n.line):
            outage_mask = rng.random(len(n.line)) < outage_p
            n.line.in_service = ~outage_mask

        # run power flow
        try:
            pp.runpp(n, numba=use_numba)
        except Exception:
            continue  # skip infeasible scenarios

        vm = n.res_bus.vm_pu.values
        p_load = n.load.groupby("bus").p_mw.sum().reindex(n.bus.index, fill_value=0).values
        alarm = ((vm < 0.95) | (vm > 1.05)).astype(int)

        all_buses.append(pd.DataFrame({
            "bus": n.bus.index.astype(int),
            "voltage": vm,
            "load_MW": p_load,
            "alarm_flag": alarm,
            "scenario": s
        }))

        all_edges.append(pd.DataFrame({
            "from_bus": n.line.from_bus.values,
            "to_bus": n.line.to_bus.values,
            "x_pu": n.line.x_ohm_per_km.values if "x_ohm_per_km" in n.line else np.nan,
            "in_service": n.line.in_service.values,
            "length_km": n.line.length_km.values if "length_km" in n.line else np.nan,
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

    bus_df.to_csv("bus_scenarios.csv", index=False)
    edge_df.to_csv("edge_scenarios.csv", index=False)
    print(f"✅ Generated {len(bus_df)} bus rows = {n_buses} buses × {n_scenarios} scenarios.")
    print(f"   Edges: {len(edge_df)} rows (~{edges_mean:.1f} per scenario, min {edges_min}, max {edges_max}).")

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