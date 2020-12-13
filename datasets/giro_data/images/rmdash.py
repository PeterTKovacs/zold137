import pandas as pd 
import os
import shutil

datasets=['giro1','giro8','giro4']
subfolders=['train', 'test', 'valid']
rootdir='.'

def removedash(src,dest,ann_file):

    images=os.listdir(src)

    annot = pd.read_csv(ann_file, header = None ,sep = ' ' )
    annot = annot.rename(columns={0: "dr1", 1: "frame", 2: "ID", 3: "X1", 4: "Y1", 5: "X2", 6: "Y2", 7: "dr2", 8: "object", 9: "dr3"})
    annot.drop(columns = ["dr1", "dr2", "dr3"], inplace = True)
    dsh=annot[annot['object']=='-']

    to_replace=set(dsh['frame'])

    for image in images:
        frame_name=(image.split('_'))[-1].split('.')[0]
        if int(frame_name) in to_replace:
        # move to dash
            shutil.move(os.path.join(src,image),dest)

            print('moved '+image)

    return len(dsh)


def move_dash(root,dataset,sfd,ann):
    for folder in sfd:
        try:
            os.mkdir(os.path.join(root,dataset,'dash',folder))
        except:
            print('already exists?')
        removedash(str(os.path.join(root,dataset,folder)),str(os.path.join(root,dataset,'dash',folder)),ann)
    return 0

for dataset in datasets:
    annfile='/zold137/datasets/giro_data/annot/'+dataset+'.txt'
    try:
        os.mkdir(os.path.join(rootdir,dataset,'dash'))
    except:
        print('already exists?')

    move_dash(rootdir,dataset,subfolders,annfile)

