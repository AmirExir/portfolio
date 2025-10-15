import pandapower.networks as pn
import pandas as pd
import numpy as np
import pandapower as pp

def build_ieee118():
    net = pn.case118()   # built-in IEEE-118 test system
    return net

def sample_scenarios(net, n_scen=50, outage_p=0.03, load_sigma=0.1, seed=42):
    rng = np.random.default_rng(seed)
    all_buses, all_edges = [], []

    for s in range(n_scen):
        n = net.deepcopy()

        # jitter loads ±10%
        if len(n.load):
            n.load["p_mw"] *= (1.0 + rng.normal(0, load_sigma, len(n.load)))

        # randomly open ~3% of lines
        if len(n.line):
            outage_mask = rng.random(len(n.line)) < outage_p
            n.line.in_service = ~outage_mask

        # run power flow
        try:
            pp.runpp(n, numba=True)
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
            "scenario": s
        }))

    bus_df = pd.concat(all_buses, ignore_index=True)
    edge_df = pd.concat(all_edges, ignore_index=True)
    bus_df.to_csv("bus_scenarios.csv", index=False)
    edge_df.to_csv("edge_scenarios.csv", index=False)
    print(f"✅ Generated {len(bus_df)} bus rows and {len(edge_df)} edges across {len(bus_df['scenario'].unique())} scenarios.")

if __name__ == "__main__":
    net = build_ieee118()
    sample_scenarios(net, n_scen=50)