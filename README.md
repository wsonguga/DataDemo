# DataDemo
Cardiorespiratory Time-series Sensor Data Simulation and Demo

## Data simulations

### SCG (Seismocardiogram)

The SCG data simulation is built on the top of NeuroKit2: https://github.com/neuropsychology/NeuroKit

Generate the simulated SCG data:
```
  python3 simscg.py
```

Visualize the saved SCG data npy file:
```
  python3 view_data.py [xxx.npy] [num_labels]
```

Perform clustering of the simulated data on quality evaluation:
```
  python3 evalscg.py
```
### BSG (Bodyseismogram)

Generate the simulated BSG data:
```
  python simbsg.py
```

You can run the test.ipynb to see the visualization of each step
