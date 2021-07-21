# Tensorflow on Mac

[![Visualize in WandB](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg)](https://wandb.ai/kad99kev/m1-benchmark)

---

**Trying to get Tensorflow working on Macs with M1 chip.**

Results compared with:
1. Mac Mini with M1 (GPU/CPU)
2. Macbook Air with Intel
3. Google Colab (Tesla P100-PCIE-16GB)

---

## How to get it working?

* For M1 chips install the [m1-requirements.txt](https://github.com/kad99kev/TFonMac/blob/master/m1-requirements.txt)
* For other devices [requirements.txt](https://github.com/kad99kev/TFonMac/blob/master/requirements.txt)
* Run ```main.py``` to start the program.
* In case you want to switch to ```GPU/CPU``` mode for the M1 chip, you can find it in utils/info.py and set ```mlcompute.set_mlc_device(device_name="...")``` to ```"gpu", "cpu" or "any"```

---

## References:

1. [How To Install TensorFlow on M1 Mac (The Easy Way)](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706)
2. [Can Apple's M1 help you train models faster & cheaper than NVIDIA's V100?](https://wandb.ai/vanpelt/m1-benchmark/reports/Can-Apple-s-M1-help-you-train-models-faster-cheaper-than-NVIDIA-s-V100---VmlldzozNTkyMzg)
3. [Getting Started with tensorflow-metal PluggableDevice](https://developer.apple.com/metal/tensorflow-plugin/)
4. [A useful thread on stackoverflow](https://stackoverflow.com/questions/67167886/make-tensorflow-use-the-gpu-on-an-arm-mac)
