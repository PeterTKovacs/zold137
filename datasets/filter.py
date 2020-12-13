import os
import shutil
import random

def clean_pics(src,sink,dataset):

    ioi={ # intervals of interest for frame numbers for given videos
        'giro1': [(682,2010),(4640,6120),(8040,9280)],
        'giro4': [(3890,4200),(8880,8970),(9985,10080),(10825,11160),(11880,12480),
                  (13035,13200),(16395,16750),(18000,18285),(19080,19440)],
        'giro8': [(315,2160)],
        'giro9': [(4275,4390)]
    }
    
    fnames=os.listdir(src)
    
    for fname in fnames:
        
        frame_no=int((fname.split('_')[1])[:-4]) # from eg. giro1_233.jpg chop off unncessary parts
        
        drop=True
        
        for interval in ioi[dataset]:
            if frame_no>=interval[0]*25/24 and frame_no<=interval[1]*25/24: # DAMN: the videos are 25 fps
                drop=False
                interval_=interval
                break
                
        if drop:
            shutil.move(os.path.join(src,fname),sink)
        else:
            print('no: %d, bounds %f; %f' % (frame_no,interval_[0]*25/24,interval_[1]*25/24))
def back_from_sink(root,dataset):

    ioi={ # intervals of interest for frame numbers for given videos
        'giro1': [(682,2010),(4640,6120),(8040,9280)],
        'giro4': [(3890,4200),(8880,8970),(9985,10080),(10825,11160),(11880,12480),
                  (13035,13200),(16395,16750),(18000,18285),(19080,19440)],
        'giro8': [(315,2160)],
        'giro9': [(4275,4390)]
    }
    
    fnames=os.listdir(os.path.join(root,'sink'))
    
    for fname in fnames:

        frame_no=int((fname.split('_')[1])[:-4]) # from eg. giro1_233.jpg chop off unncessary parts

        restore=False

        for interval in ioi[dataset]:
            if frame_no>=interval[0]*0.96 and frame_no<=interval[1]*0.96: # DAMN: the videos are 25 fps
                restore=True
                break

        if restore:
            
            decision=random.random()
            if decision>0.15:
                dest=os.path.join(root,'train')
            elif decision>0.05:
                dest=os.path.join(root,'test')
            else:
                dest=os.path.join(root,'valid')

            shutil.move(os.path.join(root,'sink',fname),dest)

