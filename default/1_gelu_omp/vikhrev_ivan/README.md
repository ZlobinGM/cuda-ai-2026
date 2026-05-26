# GELU


### Build

```
g++ -O3 -fopenmp -o gelu main.cpp gelu_omp.cpp
```
OR
```
g++ -O3 -march=native -fopenmp \
    -fopt-info-vec-optimized \
    -fopt-info-vec-missed \
    -o gelu main.cpp gelu_omp.cpp
```

### Run
```
OMP_NUM_THREADS=1 ./gelu
OMP_NUM_THREADS=2 ./gelu
OMP_NUM_THREADS=4 ./gelu
OMP_NUM_THREADS=8 ./gelu
OMP_NUM_THREADS=16 ./gelu
```

### Results
Results for 100000 elements:
* CPU 12th Gen Intel® Core™ i5-1240P × 16

```
Gelu Sequential Naive impl: 182.682 ms
Gelu OMP fast tanh 2 threads: 33.150 ms
Gelu OMP fast tanh 4 threads: 25.247 ms
Gelu OMP fast tanh 8 threads: 22.856 ms
Gelu OMP fast tanh 16 threads: 18.6468 ms
```