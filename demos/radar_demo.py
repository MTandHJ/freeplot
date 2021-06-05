import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from freeplot.base import FreePlot
from freeplot.zoo import pre_radar, pos_radar



labels = (
    "brightness", "defocus_blur", "fog", "gaussian_blur", "glass_blur", "jpeg_compression",
    "motion_blur", "saturate, snow", "speckle_noise", "contrast", "elastic_transform", "frost",
    "gaussian_noise", "impulse_noise", "pixelate", "shot_noise", "spatter", "zoom_blur", "transform", "flowSong"
)

theta = pre_radar(len(labels), frame="polygon")

# shape: 1, 1; figsize: 4, 4;
fp = FreePlot((1, 1), (4, 4), dpi=100, titles=["RADAR"], projection="radar")


data = {
    "A": np.random.rand(len(labels)),
    'B': np.random.rand(len(labels)),
    'C': np.random.rand(len(labels))
}

pos_radar(data, labels, fp)

fp[0, 0].legend()

# fp.savefig("radar_demo.pdf", format="pdf", tight_layout=True)
plt.show()

