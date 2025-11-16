## Project layout

```
.
├── d3pm/
│   ├── __init__.py
│   ├── data.py          # Text8 utilities and dataset class
│   ├── diffusion.py     # Discrete diffusion forward process helpers
│   └── model.py         # Transformer denoiser + sinusoidal timestep encodings
├── train_text8.py       # CLI entry-point for training
├── sample_text.py       # Simple ancestral sampling loop
├── requirements.txt
└── README.md
```


