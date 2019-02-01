import imgaug as ia
import matplotlib.pylab as plt
from imgaug import augmenters as iaa
import numpy as np


plt.figure(figsize=(10,10),dpi=(150))
def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    ravel= axeslist.ravel()
    for ind,img in enumerate(figures):
        ravel[ind].imshow(img, cmap=plt.jet())
        ravel[ind].set_title(" %2d "%(ind))
        ravel[ind].set_axis_off()
    plt.tight_layout() # optional



ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
images = np.array(
    [ia.quokka(size=(128, 128)) for _ in range(12)],
    dtype=np.uint8
)
# images =ia.quokka(size=(64, 64))
seq = iaa.Sequential([
    # #iaa.Fliplr(0.5), # horizontal flips
    # iaa.Crop(percent=(0.0, 0.15)), # random crops
    # # Small gaussian blur with random sigma between 0 and 0.5.
    # # But we only blur about 50% of all images.
    # iaa.Sometimes(0.5,
    #     iaa.GaussianBlur(sigma=(0.0 , 0.5))
    # ),
    # Strengthen or weaken the contrast in each image.
    #iaa.ContrastNormalization((1.5, 1.5)),
    iaa.AdditiveGaussianNoise(scale=0.03*255, per_channel=0.5)
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    # iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # # Apply affine transformations to each image.
    # # Scale/zoom them, translate/move them, rotate them and shear them.
    # iaa.Affine(
    #     #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #     scale=(0.6, 1.6),
    #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #     rotate=(-10, 10),
    #     shear=(-4, 4)
    # )
], random_order=True) # apply augmenters in random order
seq.to_deterministic()
images_aug = seq.augment_images(images)


#one_image=images_aug[0]
#plt.imshow(one_image)
plot_figures(images_aug,3,4)

plt.show()

