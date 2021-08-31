# Data

Each run contains data saved into two files `Energy_data.npz` and `Measurement_data.npz` which contain data and information about the quantum simulation runs.

`Energy_data.npz`: Contains information from the variational quantum simulations including the final energy, the

`Measurement_data.npz`: Contains the nearly-diagonal measurement data taken on the final optimized VQE result. These measurements are used in NEM during the NQST phase.

**Example:** To load both the VQE energy and measurement data files:

```python
vqe_data_dir= sys.path[0] + "/../data/H2molecule/DepolarizingNoise/BL_0.2/Run0/"

vqe_results = np.load(data_file_path + "Energy_data.npz", allow_pickle=True).get('arr_0')
tomography_measurements = np.load(data_file_path + "Measurement_data.npz", allow_pickle=True).get('arr_0')

# Extract results from the loaded dictionaries
energy_vqe = vqe_results.item()['energy_vqe']
fidelity_vqe = vqe_results.item()['dm_fidelity']
```
