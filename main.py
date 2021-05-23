from model import *
from data import *

import skimage.io as io
import numpy as np

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')


myGene = trainGenerator(4,'ASOCA2020Data','0_Train','0_Mask',data_gen_args,save_to_dir=None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

testGene = testGenerator('0_Test/')
results = model.predict_generator(testGene,30,verbose=1)

saveResult("0_Result/",results)

# evalResult()

