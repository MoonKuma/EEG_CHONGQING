# Preprocessing pipeline

### Resting states

##### Goal

- Averaged EEG Morlet power data on 5 channel areas (F/C/T/P/O) times 5 frequency bands(2.5, 5.0, 10.0, 17., 35.)
- Normalized across channel areas and then saved with subject number as index 

##### Pipeline

- Load : .cnt -> ram

- down sample: 250 Hz

- Filter: (1~50), band

- ICA to repair eye movement : 25 components

- Randomly Pick events window:  continuously 1 seconds randomly located in every 2 seconds window

- Epoch & reject : >25

- Morlet : power only,  freqs: [2.5, 5.0, 10.0, 17., 35.],  n_cycles = freqs / 2.

- Average across time window : no baseline correction

- Average across channel areas : F/C/T/P/O

- Normalize across channel areas : F/C/T/P/O, means only the relative activity [this is optional]

-  Save result json as 'eeg_rest_sub5.txt'

  ```
  {
      'index': 'sub2'
      'type': 'rest'
      'data': numpy.ndarray (5,5)
      'norm_data': numpy.ndarray (5,5)
  }
  ```


### Pain Neutral