import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd

def sample_scenario(base_net, rng):
    net = base_net.deepcopy()
    # Randomly scale loads
    for ld in net.load.index:
        net.load.at[ld, "p_mw"] *= rng.uniform(0.7, 1.3)
        net.load.at[ld, "q_mvar"] *= rng.uniform(0.7, 1.3)
    # Random N-1 line outage (simulate contingency)
    if rng.random() < 0.5:
        out = rng.choice(net.line.index)
        net.line.at[out, "in_service"] = False
    try:
        pp.runpp(net, enforce_q_lims=True, init="results")
    except:
        return None, None
    vm = net.res_bus.vm_pu.values
    p_load = np.zeros(len(net.bus))
    for _, row in net.load.iterrows():
        p_load[int(row.bus)] += float(row.p_mw)
    alarms = ((vm < 0.95) | (vm > 1.05)).astype(int)
    df = pd.DataFrame({
        "bus": net.bus.index,
        "voltage": vm,
        "load_MW": p_load,
        "alarm_flag": alarms
    })
    active_lines = net.line[net.line.in_service]
    edge_df = pd.DataFrame({
        "from_bus": active_lines.from_bus.values,
        "to_bus": active_lines.to_bus.values
    })
    return df, edge_df

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    net = pn.case14()
    bus_data = []
    edge_data = []
    for i in range(1000):
        d, e = sample_scenario(net, rng)
        if d is not None and e is not None:
            d["scenario"] = i
            e["scenario"] = i
            bus_data.append(d)
            edge_data.append(e)
    df_all_bus = pd.concat(bus_data, ignore_index=True)
    df_all_edge = pd.concat(edge_data, ignore_index=True)
    df_all_bus.to_csv("bus_scenarios.csv", index=False)
    df_all_edge.to_csv("edge_scenarios.csv", index=False)
    print("✅ Saved bus dataset:", df_all_bus.shape)
    print("✅ Saved edge dataset:", df_all_edge.shape)
    print(df_all_bus.head())