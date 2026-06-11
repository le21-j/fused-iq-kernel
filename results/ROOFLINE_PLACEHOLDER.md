# Roofline Plot

results/roofline.png is generated only on the GPU host by running bench/make_roofline.py.
The script reads bench.json (produced by benchmark.py --json results/bench.json), accepts
the exact GPU model string via --gpu (which must match the output of nvidia-smi on the target
machine and is embedded verbatim in the plot title), and accepts peak memory bandwidth and
peak FP32 compute via --peak-bw-gbs and --peak-tflops respectively. These ceiling values are
supplied by the operator from the hardware datasheet for whatever GPU nvidia-smi reports; they
are never hard-coded inside the script. The plot is written to results/roofline.png.
