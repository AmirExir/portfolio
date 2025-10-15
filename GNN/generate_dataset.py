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
        return None
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
    return df

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    net = pn.case14()
    data = []
    for i in range(1000):
        d = sample_scenario(net, rng)
        if d is not None:
            d["scenario"] = i
            data.append(d)
    df_all = pd.concat(data, ignore_index=True)
    df_all.to_csv("bus_scenarios.csv", index=False)
    print("✅ Saved dataset:", df_all.shape)
    print(df_all.head())