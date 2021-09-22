# -*- coding:utf-8 -*-

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras import initializers
import cv2
from hs_data_v3 import *

# np.random.seed(777) # for reproducibility
# initializers.he_normal(seed=777)


class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512, custom_name='/hs_v3'):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.custom_name = custom_name

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_guide_train, imgs_mask_train = mydata.load_train_data()
        imgs_test, imgs_guide_test = mydata.load_test_data()
        return imgs_train, imgs_guide_train, imgs_mask_train, imgs_test, imgs_guide_test

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 2))
        guides = Input((self.img_rows, self.img_cols, 1))
        print(inputs.shape)
        print(guides.shape)

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        concatenate6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concatenate6) #0
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        concatenate7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concatenate7) #1
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        concatenate8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concatenate8) #2
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        concatenate9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concatenate9) #3
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9) # 3 2

        #tobe
        # inputs_edge = inputs[:,:,:,-1] #.reshape(5,5,1)
        # inputs_edge = core.Reshape((512,512,1))(inputs_edge)
        concatenate10 = concatenate([conv9, guides], axis=3)
        conv10 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concatenate10) #4
        conv10 = Conv2D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv10)


        #asis
        #conv10 = Conv2D(1, 1, activation='sigmoid')(conv9) #hs 1로수정 softmax sigmoid
        
        model = Model([inputs, guides], conv10)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy']) # categorical_crossentropy binary_crossentropy

        return model

    def train(self):
        print("loading data")
        imgs_train, imgs_guide_train, imgs_mask_train, imgs_test, imgs_guide_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        model_checkpoint = ModelCheckpoint('./weights' + self.custom_name + '_unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit((imgs_train, imgs_guide_train), imgs_mask_train, batch_size=1, epochs=50, verbose=1,
                  validation_split=0.1, shuffle=True, callbacks=[model_checkpoint])

        print('predict test data')
        print('test candidate data : {}'.format(imgs_test.shape))
        imgs_mask_test = model.predict((imgs_test, imgs_guide_test), batch_size=1, verbose=1)
        print('test result data : {}'.format(imgs_mask_test.shape))
        np.save('./results' + self.custom_name + '_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("array to image")
        imgs = np.load('./results' + self.custom_name + '_mask_test.npy')
        piclist = []
        for line in open("./results" + self.custom_name + ".txt"):
            line = line.strip()
            picname = line.split('/')[-1]
            piclist.append(picname)
        for i in range(imgs.shape[0]):
            path = "./results/" + piclist[i]
            img = np.zeros((imgs.shape[1], imgs.shape[2], 3), dtype=np.uint8)
            for k in range(len(img)):
                for j in range(len(img[k])):
                    # 0화이트 배경 1블랙 엣지
                    if float(imgs[i][k][j]) >= 0.5:
                        img[k][j]=[255,255,255]
                    elif float(imgs[i][k][j]) < 0.5:
                        img[k][j]=[0,0,0]


                    # num = np.argmax(imgs[i][k][j])
                    # if num == 0:
                    #     #img[k][j] = [128, 128, 128]
                    #     img[k][j] = [0, 0, 0]
                    # elif num == 1:
                    #     #img[k][j] = [128, 0, 0]
                    #     #img[k][j] = [255, 0, 0]
                    #     img[k][j] = [255, 255, 255]
                    # elif num == 2:
                    #     #img[k][j] = [192, 192, 128]
                    #     img[k][j] = [255, 255, 255]

            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(path, img)


if __name__ == '__main__':


    mydata = dataProcess(512, 512)
    mydata.create_train_data()
    mydata.create_test_data()

    myunet = myUnet()
    model = myunet.get_unet()
    model.summary()
    # plot_model(model, to_file='model.png')
    myunet.train()
    myunet.save_img()
