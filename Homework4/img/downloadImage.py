from skimage import data, io, color, img_as_ubyte
imgs = [data.camera(), data.coins(), data.moon(), data.astronaut(), data.coffee()]
for i, im in enumerate(imgs, 1):
    if im.ndim == 3:                
        im = img_as_ubyte(color.rgb2gray(im))
    io.imsave(f"img{i}.png", im) 