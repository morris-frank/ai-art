# vqgan shopngle
python mmv.py shpongle.1280.jpg -music Shpongolese\ Spoken\ Here.wav -script Shpongle.csv -lr_warmup 0.005 -size 512 512


# vqgan
export name="Eating\ Hooks"; ssh ws4 "cd private_code/aiart/vqgan_clip; ffmpeg -framerate 30 -pattern_type glob -i 'upsampled_frames/*.jpg'  -i ../inputs/${name}.wav -vf scale=1920:1080 -shortest -y ${name}.mp4"; scp ws4:"private_code/aiart/vqgan_clip/${name}.mp4" .
# m_analyz
export name="Nothing\ Is\ Something\ Worth\ Doing"; ssh ws4 "cd private_code/aiart; ffmpeg -framerate 25 -pattern_type glob -i 'frames/*.jpg'  -i inputs/${name}.wav -shortest -y ${name}.mp4"; scp ws4:"private_code/aiart/${name}.mp4" .
# local
export name="Nothing Is Something Worth Doing"; ffmpeg -framerate 25 -pattern_type glob -i 'frames/*.jpg'  -i "/Users/mfr/Downloads/${name}.wav" -shortest -y "${name}.mp4"