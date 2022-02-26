# DataDemo
Time Series Data Simulation and Analytics Demo

## Data simulations

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


