# Preprocessing pipeline

### Resting states

##### Goal

- Averaged EEG Morlet power data on 5 channel areas (F/C/T/P/O) times 5 frequency bands(2.5, 5.0, 10.0, 17., 35.)
- Normalized across channel areas and then saved with subject number as index 

##### Pipeline

- Load : .cnt -> ram
- down sample: 250 Hz
- Filter: (1~50)
- ICA to repair eye movement : 25 components
- Randomly Pick events window:  continuously 1 seconds randomly located in every 2 seconds window
- Epoch & reject : >25
- Morlet : power only,  freqs: [2.5, 5.0, 10.0, 17., 35.],  n_cycles = freqs / 2.
- Average across time window : -1~1, no baseline correction
- Average across channel areas : F/C/T/P/O
- Normalize across channel areas : F/C/T/P/O, means only the relative activity [this is optional]
- Save result

##### Result

- n subjects * 50 features
  - 50 features : normalized 5 channels * 5 frequencies, unnormalized 25 features  

### Pain Neutral

##### Goal

- Averaged EEG Morlet power data on 5 channel areas (F/C/T/P/O) times 5 frequency bands(2.5, 5.0, 10.0, 17., 35.), 5 time windows(0~200, 200~400, 400~600, 600~800, 800~1000) and 4 type of events (Pain_F/Pain_M/Neutral_F/Neutral_M)
- Auto-detected peak latency (1) and peak amplitude (ERP) averaged across 5 channel areas in 2 time windows (0~200, 200~400)

##### Pipeline

- Load
- Filter
- ICA
- Epoch by events type & Reject
- Down sample (conflict may occur if down sampling raw data) 
- Morlet & Average across time windows, channels, baseline (-1~0)
- Get ERP peak latency/amplitude & Average across  time windows, channels, baseline (-1~0)
- Save original/normalized data