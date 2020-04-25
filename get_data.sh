# from https://github.com/aditya12agd5/divcolor
wget http://vision.cs.illinois.edu/projects/divcolor/data.zip
unzip -qq data.zip
rm data.zip
wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
tar -xzf lfw-deepfunneled.tgz
mv lfw-deepfunneled data/lfw_images
rm lfw-deepfunneled.tgz
