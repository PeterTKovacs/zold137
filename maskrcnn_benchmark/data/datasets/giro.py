import pandas as pd
import torch
from PIL import Image
import os
from maskrcnn_benchmark.structures.bounding_box import BoxList

class giro(object):
    def __init__(self,ann_file,root,transforms=None): #the size value controls how big the pictures we want to be
      
        self.root= root # directory for pictures
        self.img_fnames=list(os.listdir(root))
        self.index_to_fname={idx:item for idx,item in enumerate(self.img_fnames)}
        
        annot = pd.read_csv(ann_file, header = None ,sep = ' ' )
        annot = annot.rename(columns={0: "dr1", 1: "frame", 2: "ID", 3: "X1", 4: "Y1", 5: "X2", 6: "Y2", 7: "dr2", 8: "object", 9: "dr3"})
        annot.drop(columns = ["dr1", "dr2", "dr3"], inplace = True)
 #       annot=annot[annot['object']!='-']

        self.boxes=annot
        self.transforms=transforms
        self.object_to_cat={
                            '1F': 'Front View',
                            '1B': 'Back View',
                            '1L': 'Left View',
                            '1R': 'Right View',
                            '2': ' Bicycle Crowd',
                            '5H': 'High-Density Human Crowd',
                            '5L': 'Low-Density Human Crowd',
                            '0': 'irrelevant TV graphics',
                           '00':'__background'}
        self.object_to_id={'1F': 8,
                            '1B': 1,
                            '1L': 2,
                            '1R': 3,
                            '2': 4,
                            '5H': 5,
                            '5L': 6,
                            '0': 7,
                             '00':0}

    def __len__(self):
        return len(self.img_fnames)

    def get_frame_no(self,fname):
        
        tmp=fname.split('_')
        
        if tmp[1][-4:]=='.jpg':
            return int(tmp[1][:-4])
        else:
            print('invalid filename')
            return -1

    def __getitem__(self, idx): 
        
        # load the image as a PIL Image
        # DAMN! annotations have been prepared in 1920x1080 so we must resize pics to this size
        image = Image.open(os.path.join(self.root,self.index_to_fname[idx])).resize((1920,1080))

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        
        M_=self.boxes[self.boxes.frame==self.get_frame_no(self.index_to_fname[idx])]
        m__=M_[["X1", "Y1", "X2", "Y2"]].values.tolist()
            
        labels = torch.tensor([self.object_to_id[obj] for obj in list(M_['object'])])
        boxlist = BoxList(m__, image.size, mode="xyxy")
            
        # add the labels to the boxlist
        
        boxlist.add_field("labels", labels)
        
        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)
        return image, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        
        image = Image.open(os.path.join(self.root,self.index_to_fname[idx]))

        width, height= image.size
        return {"height": height, "width": width}

