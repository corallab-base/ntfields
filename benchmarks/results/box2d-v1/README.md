### box2d-v1 (test)

| problem | entry | seeds | success % | SPL | cost ratio | T error % | plan ms | train min |
|---|---|---|---|---|---|---|---|---|
| clutter | euclidean+gradient_bidirectional | 3 | 21.5 ± 0.0 | 0.189 ± 0.000 | 1.176 ± 0.000 | 13.9 ± 0.1 | 10 ± 0 | 0.1 ± 0.1 |
| clutter | euclidean+mppi | 3 | 22.0 ± 0.9 | 0.191 ± 0.006 | 1.196 ± 0.013 | 13.9 ± 0.1 | 47 ± 3 | 0.1 ± 0.1 |
| clutter | pntfields_2d+gradient_bidirectional | 3 | 85.2 ± 8.6 | 0.807 ± 0.081 | 1.061 ± 0.002 | 12.4 ± 1.0 | 56 ± 1 | 10.1 ± 1.0 |
| clutter | pntfields_2d+mppi | 3 | 86.2 ± 6.4 | 0.821 ± 0.061 | 1.053 ± 0.005 | 12.4 ± 1.0 | 116 ± 10 | 10.1 ± 1.0 |
| clutter | td_ntfields_2d+gradient_bidirectional | 3 | 85.8 ± 4.0 | 0.815 ± 0.037 | 1.068 ± 0.015 | 6.4 ± 0.5 | 183 ± 10 | 2.8 ± 0.0 |
| clutter | td_ntfields_2d+mppi | 3 | 87.0 ± 1.3 | 0.845 ± 0.013 | 1.031 ± 0.003 | 6.4 ± 0.5 | 572 ± 7 | 2.8 ± 0.0 |
| empty | euclidean+gradient_bidirectional | 3 | 100.0 ± 0.0 | 1.000 ± 0.000 | 0.999 ± 0.000 | 0.0 ± 0.0 | 10 ± 0 | 0.0 ± 0.0 |
| empty | euclidean+mppi | 3 | 100.0 ± 0.0 | 0.998 ± 0.000 | 1.002 ± 0.000 | 0.0 ± 0.0 | 45 ± 4 | 0.0 ± 0.0 |
| empty | pntfields_2d+gradient_bidirectional | 3 | 100.0 ± 0.0 | 1.000 ± 0.000 | 0.999 ± 0.000 | 0.0 ± 0.0 | 41 ± 1 | 9.0 ± 1.8 |
| empty | pntfields_2d+mppi | 3 | 100.0 ± 0.0 | 0.998 ± 0.000 | 1.002 ± 0.000 | 0.0 ± 0.0 | 90 ± 1 | 9.0 ± 1.8 |
| empty | td_ntfields_2d+gradient_bidirectional | 3 | 100.0 ± 0.0 | 0.999 ± 0.000 | 1.001 ± 0.000 | 0.8 ± 0.2 | 90 ± 2 | 2.7 ± 0.3 |
| empty | td_ntfields_2d+mppi | 3 | 100.0 ± 0.0 | 0.997 ± 0.000 | 1.003 ± 0.000 | 0.8 ± 0.2 | 229 ± 4 | 2.7 ± 0.3 |
| one_sphere | euclidean+gradient_bidirectional | 3 | 63.0 ± 0.0 | 0.604 ± 0.000 | 1.059 ± 0.000 | 3.9 ± 0.1 | 10 ± 0 | 0.0 ± 0.0 |
| one_sphere | euclidean+mppi | 3 | 62.8 ± 0.3 | 0.602 ± 0.003 | 1.059 ± 0.002 | 3.9 ± 0.1 | 48 ± 3 | 0.0 ± 0.0 |
| one_sphere | pntfields_2d+gradient_bidirectional | 3 | 100.0 ± 0.0 | 0.995 ± 0.000 | 1.005 ± 0.000 | 5.0 ± 1.2 | 45 ± 1 | 36.8 ± 7.2 |
| one_sphere | pntfields_2d+mppi | 3 | 99.8 ± 0.3 | 0.992 ± 0.002 | 1.006 ± 0.001 | 5.0 ± 1.2 | 97 ± 1 | 36.8 ± 7.2 |
| one_sphere | td_ntfields_2d+gradient_bidirectional | 3 | 99.5 ± 0.5 | 0.972 ± 0.005 | 1.026 ± 0.005 | 4.1 ± 0.7 | 116 ± 17 | 2.9 ± 0.0 |
| one_sphere | td_ntfields_2d+mppi | 3 | 99.7 ± 0.6 | 0.983 ± 0.006 | 1.014 ± 0.001 | 4.1 ± 0.7 | 488 ± 4 | 2.9 ± 0.0 |
| thick_walls | euclidean+gradient_bidirectional | 3 | 20.0 ± 0.0 | 0.195 ± 0.000 | 1.027 ± 0.000 | 38.2 ± 0.1 | 10 ± 0 | 0.0 ± 0.0 |
| thick_walls | euclidean+mppi | 3 | 20.0 ± 0.0 | 0.195 ± 0.000 | 1.030 ± 0.002 | 38.2 ± 0.1 | 49 ± 3 | 0.0 ± 0.0 |
| thick_walls | pntfields_2d+gradient_bidirectional | 3 | 90.5 ± 4.4 | 0.874 ± 0.049 | 1.038 ± 0.010 | 13.8 ± 9.3 | 114 ± 27 | 17.0 ± 2.4 |
| thick_walls | pntfields_2d+mppi | 3 | 77.8 ± 1.0 | 0.755 ± 0.018 | 1.033 ± 0.013 | 13.8 ± 9.3 | 170 ± 32 | 17.0 ± 2.4 |
| thick_walls | td_ntfields_2d+gradient_bidirectional | 3 | 99.5 ± 0.0 | 0.977 ± 0.002 | 1.019 ± 0.002 | 4.4 ± 0.2 | 226 ± 26 | 2.7 ± 0.2 |
| thick_walls | td_ntfields_2d+mppi | 3 | 77.8 ± 0.3 | 0.766 ± 0.004 | 1.016 ± 0.001 | 4.4 ± 0.2 | 794 ± 1 | 2.7 ± 0.2 |
| walls | euclidean+gradient_bidirectional | 3 | 19.0 ± 0.0 | 0.185 ± 0.000 | 1.043 ± 0.000 | 38.2 ± 0.1 | 10 ± 0 | 0.0 ± 0.0 |
| walls | euclidean+mppi | 3 | 19.0 ± 0.0 | 0.184 ± 0.000 | 1.047 ± 0.005 | 38.2 ± 0.1 | 46 ± 4 | 0.0 ± 0.0 |
| walls | pntfields_2d+gradient_bidirectional | 3 | 64.3 ± 4.5 | 0.630 ± 0.047 | 1.021 ± 0.006 | 8.2 ± 2.2 | 76 ± 2 | 20.6 ± 0.9 |
| walls | pntfields_2d+mppi | 3 | 64.3 ± 4.2 | 0.627 ± 0.043 | 1.027 ± 0.008 | 8.2 ± 2.2 | 140 ± 8 | 20.6 ± 0.9 |
| walls | td_ntfields_2d+gradient_bidirectional | 3 | 99.8 ± 0.3 | 0.981 ± 0.004 | 1.018 ± 0.002 | 4.1 ± 0.5 | 205 ± 30 | 2.8 ± 0.0 |
| walls | td_ntfields_2d+mppi | 3 | 86.0 ± 0.5 | 0.848 ± 0.006 | 1.015 ± 0.002 | 4.1 ± 0.5 | 750 ± 58 | 2.8 ± 0.0 |
| all | euclidean+gradient_bidirectional | 3 | 44.7 ± 0.0 | 0.435 ± 0.000 | 1.061 ± 0.000 | 18.8 ± 0.0 | 10 ± 0 | 0.0 ± 0.0 |
| all | euclidean+mppi | 3 | 44.8 ± 0.2 | 0.434 ± 0.001 | 1.067 ± 0.002 | 18.8 ± 0.0 | 47 ± 0 | 0.0 ± 0.0 |
| all | pntfields_2d+gradient_bidirectional | 3 | 88.0 ± 1.7 | 0.861 ± 0.015 | 1.025 ± 0.002 | 7.9 ± 1.3 | 66 ± 5 | 18.7 ± 1.1 |
| all | pntfields_2d+mppi | 3 | 85.6 ± 1.9 | 0.839 ± 0.018 | 1.024 ± 0.004 | 7.9 ± 1.3 | 123 ± 9 | 18.7 ± 1.1 |
| all | td_ntfields_2d+gradient_bidirectional | 3 | 96.9 ± 0.7 | 0.949 ± 0.007 | 1.026 ± 0.003 | 4.0 ± 0.1 | 164 ± 4 | 2.8 ± 0.1 |
| all | td_ntfields_2d+mppi | 3 | 90.1 ± 0.3 | 0.888 ± 0.003 | 1.016 ± 0.000 | 4.0 ± 0.1 | 567 ± 10 | 2.8 ± 0.1 |

