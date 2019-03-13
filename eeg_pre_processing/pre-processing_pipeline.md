# Preprocessing pipeline

### Resting states

##### Goal

- EEG : Averaged EEG Morlet power data on 5 channel areas (F/C/T/P/O) times 5 frequency bands(2.5, 5.0, 10.0, 17., 35.)

##### Pipeline

- Preprocessing
  - Load cnt 
  - down sample: 250 Hz
  - Filter: (1~50) band filter
  - ICA to repair eye movement : 25 components, 'fastica'
  - Randomly Pick events window:  continuously 2 seconds randomly located in every 4 seconds window
  - Epoch(-1~1s) & Reject : >25
  - Morlet : power only,  freqs: [2.5, 5.0, 10.0, 17., 35.],  n_cycles = freqs / 2.
- Time window data
  - Average across time window : -1~1, no baseline correction
  - Average across channel areas : F/C/T/P/O
  - Normalize (L2) across channel areas : F/C/T/P/O, means only the relative activity [optional]

##### Result

- n subjects * 50 features
  - 50 features 
    - normalized 5 channels * 5 frequencies
    - unnormalized 25 features  



### Pain Neutral

##### Events (4 conditions)

- Pain/Neutral * Male/Female

##### Goal

- EEG : Averaged EEG Morlet power data on 5 channel areas (F/C/T/P/O) times 5 frequency bands(2.5, 5.0, 10.0, 17., 35.) in 5 time windows(0~200, 200~400, 400~600, 600~800, 800~1000ms)  for all 4 conditions
- ERP-latency : Auto-detected peak latency (1) in 2 time windows (50~200, 200~400ms) for all 4 conditions
- ERP-amplitude : Peak amplitude averaged across 5 channel areas(F/C/T/P/O)  in 2 time windows (50~200, 200~400ms)

##### Pipeline

- Preprocessing general
  - Load cnt
  - Filter:  (1~50) band filter
  - ICA:  25 components, 'fastica'
  - Epoch for ERP
    - Epoch (-0.5,1s) by events type & Reject>25.
    - Down sample (conflict of events may occur if down sampling raw data)
    - Average epochs -> ERP data
  - Epoch for  EEG
    - Epoch (-1,1s) by events type & Reject>25.
    - Morlet: power only, freqs: [2.5, 5.0, 10.0, 17., 35.],  n_cycles = freqs / 2.
- Time window ERP
  - Baseline correction: (-0.5,0)
  - Get peak latency in time windows (50~200, 200~400ms) for all 4 conditions
  - Get peak amplitude in +- 10ms around each peaks
  - Average amplitude across channels : F/C/T/P/O
  - Normalize (L2) across channel areas : F/C/T/P/O [optional]
- Time window EEG
  - Baseline correction: (-1,0)
  - Average across time window : 0~200, 200~400, 400~600, 600~800, 800~1000ms
  - Average across channel areas : F/C/T/P/O
  - Normalize (L2) [optional]

##### Result

- n subjects * ? features
  - 50 features : normalized 5 channels * 5 frequencies, unnormalized 25 features  