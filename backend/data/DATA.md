Each row in backend/data/vessel_position_metrics.csv is one vessel (identified by its MMSI). The three numeric columns are mean-squared errors the LSTM produced on that vessel’s holdout set:

- mse_latitude – average squared difference between the predicted latitude and the true latitude, measured in degrees². Small numbers (e.g. 1e-08) mean the model’s latitude predictions are within a few micro-degrees. Larger values (0.1+) mean the model is off by several hundred meters or more.

- mse_longitude – same idea but for longitude, also in degrees². Because longitude degrees shrink by cos(latitude), an error of 1e-08 degrees is still only a few centimeters near the equator, but tens of meters near the poles.

- mse_mean – simple average of the previous two columns: (mse_latitude + mse_longitude) / 2. It’s a single quality score per vessel, so you can sort “best to worst.”

1. Take the square root to get the root mean squared error in degrees (rmse = sqrt(mse)).
2. Convert that to meters: meters ≈ rmse * 111_320 for latitude; for longitude multiply further by cos(latitude).
3. Example (mmsi 205086000):
mse_latitude ≈ 1.20e-08 → rmse_lat ≈ 0.000109° → about 12 m north–south.
mse_longitude ≈ 7.1e-10 → rmse_lon ≈ 2.7e-05°. At 45° latitude that’s ≈ 2 m east–west.
mse_mean ≈ 6.33e-09 just summarizes the two.

So, rows with numbers around 1e-08 correspond to predictions within tens of meters. Rows with values like 0.4–0.6 mean the model is missing by ~0.6° (tens of kilometers) and should be treated as “badly predicted” vessels.