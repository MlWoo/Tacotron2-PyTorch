# Tacotron2-PyTorch
PyTorch implementation of DeepMind's Tacotron-2. A deep neural network architecture described in this paper: [Natural TTS: synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/pdf/1712.05884.pdf)  

A whole TTS system is consisted of front-end and back-end. [Tacotron 2](https://arxiv.org/pdf/1712.05884.pdf) and [deep voice 3](https://arxiv.org/abs/1710.07654) are excellent front-ends. And [WaveNet](https://arxiv.org/abs/1609.03499) and [WaveGlow](https://arxiv.org/abs/1811.00002) are that back-ends.

I decide to rewrite the Tacotron 2 although (Rayhane-mamah)[https://github.com/Rayhane-mamah] and [NVidia](https://github.com/NVIDIA/) have implemented it. Because I wish I could master and implement the attention mechanism by this project.And many beginners could benifit from the conciseness to understand Tacotron 2.

It aims to cooperate with the my single-Gaussian wavenet(including autoregressive wavenet and parallel wavenet) and waveglow
which are realized with PyTorch.

## Data preprocess  
Data preprocessing is borrowed from the (Tacotron-2)[https://github.com/Rayhane-mamah/Tacotron-2] implemented with Tensorflow. It is adapted slightly to meet the requirements of my self-made mandarin datasets. You should specific the index txt and output directory like the script I provides.
```bash
#!/bin/bash
python preprocess.py --dataset=mandarin \
                     --input_txt='metadata.csv' \
	             --base_dir=./ \
		     --output_dir='dataset_mandarin'
```

## Result
The alignment at 20710 step with 32 batch size  
![](/images/20710_alignment.png "The alignment at 20710 step with 32 batch size")
