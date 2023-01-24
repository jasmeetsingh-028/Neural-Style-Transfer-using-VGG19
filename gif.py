import glob
from PIL import Image



def make_gif(frame_folder):
    frames = []
    for f in glob.iglob(frame_folder+'/*'):
        frames.append((Image.open(f)))
    frame_one = frames[0]
    frame_one.save("c1 c2.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
    
if __name__ == "__main__":
    make_gif("generated/c1 c2")