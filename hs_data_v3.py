# -*- coding:utf-8 -*-

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob


class dataProcess(object):
    def __init__(self, out_rows, out_cols, train_path="PPtest/train", train_label="PPtest/trainannot",
                 guide_path = 'PPtest/guide', custom_name = "/hs_v3", # / 앞에 붙어야하고 뒤에 _ 붙음
                 test_path="PPtest/test", test_label='PPtest/testannot', npy_path="./npydata", img_type="jpg"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.train_path = train_path
        self.train_label = train_label
        self.img_type = img_type
        #self.val_path = val_path
        #self.val_label = val_label
        self.test_path = test_path
        self.test_label = test_label
        self.npy_path = npy_path
        self.custom_name = custom_name
        self.guide_path = guide_path

#     def label2class(self, label):
#         x = np.zeros([self.out_rows, self.out_cols, 12])
#         for i in range(self.out_rows):
#             for j in range(self.out_cols):
#                 x[i, j, int(label[i][j])] = 1  # 3차원 m 지점에서는 1번째 m 클래스에 속한다.
#         return x
    def label2class(self, label):
        x = np.zeros([self.out_rows, self.out_cols])
        black = np.array((0,0,0)) # 외벽 1 / 엣지
        #blue = np.array((0,0,255)) # 창문 2
        white = np.array((255,255,255)) # 배경 3 / 나머지

        for i in range(self.out_rows):
            for j in range(self.out_cols):
                #x[i, j, int(label[i][j])] = 1  # 3차원 m 지점에서는 1번째 m 클래스에 속한다.
                temp = (np.linalg.norm(label[i][j]-white), np.linalg.norm(label[i][j]-black))
                temp_idx = temp.index(min(temp))
                x[i, j] = temp_idx # 0화이트 배경 1블랙 엣지

        return x



    def create_train_data(self):
        i = 0
        print('Creating training images...')
        imgs0 = sorted(glob.glob(self.train_path+"/*."+self.img_type))
        imgs = imgs0 #+ imgs1
        imgs = [x.replace('\\','/') for x in imgs] #hs

        labels0 = sorted(glob.glob(self.train_label+"/*."+self.img_type))
        labels = labels0 #+ labels1
        labels = [x.replace('\\','/') for x in labels] #hs

        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols,2), dtype=np.uint8)
        guidedatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(labels), self.out_rows, self.out_cols,1), dtype=np.uint8)
        print(len(imgs), len(labels))

        for x in range(len(imgs)):
            imgpath = imgs[x]
            labelpath = labels[x]

            img = load_img(imgpath, grayscale=True, target_size=[512, 512])
            label = load_img(labelpath, grayscale=True, target_size=[512, 512])
            guide = load_img(self.guide_path + '/' + imgs[i].split('/')[-1], grayscale=True, target_size=[512, 512])

            img = img_to_array(img)
            label = img_to_array(label)
            guide = img_to_array(guide)

            imgdatas[i] = np.concatenate((img, guide), axis=2) # imgdatas[i] = img
            imglabels[i] = label
            guidedatas[i] = guide

            if i % 60 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1

        print('loading done')
        np.save(self.npy_path + self.custom_name + '_train.npy', imgdatas)
        np.save(self.npy_path + self.custom_name + '_guide_train.npy', guidedatas)
        np.save(self.npy_path + self.custom_name + '_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def create_test_data(self):
        i = 0
        print('Creating test images...')
        imgs = glob.glob(self.test_path + "/*." + self.img_type) #hs
        imgs = [x.replace('\\','/') for x in imgs]

        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols,2), dtype=np.uint8)
        guidedatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        testpathlist = []

        for imgname in imgs:
            testpath = imgname
            testpathlist.append(testpath)
            img = load_img(testpath, grayscale=True, target_size=[512, 512])
            guide = load_img(self.guide_path + '/' + imgs[i].split('/')[-1], grayscale=True, target_size=[512, 512])

            img = img_to_array(img)
            guide = img_to_array(guide)

            imgdatas[i] = np.concatenate((img, guide), axis=2) # imgdatas[i] = img
            guidedatas[i] = guide
            i += 1

        txtname = './results' + self.custom_name + '.txt'
        with open(txtname, 'w') as f:
            for i in range(len(testpathlist)):
                f.writelines(testpathlist[i] + '\n')
        print('loading done')
        np.save(self.npy_path + self.custom_name +'_test.npy', imgdatas)
        np.save(self.npy_path + self.custom_name + '_guide_test.npy', guidedatas)
        print('Saving to imgs_test.npy files done.')

    def load_train_data(self):
        print('load train images...')
        imgs_train = np.load(self.npy_path + self.custom_name + "_train.npy")
        imgs_guide_train = np.load(self.npy_path + self.custom_name + "_guide_train.npy")
        imgs_mask_train = np.load(self.npy_path + self.custom_name + "_mask_train.npy")

        imgs_train = imgs_train.astype('float32')
        imgs_guide_train = imgs_guide_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_guide_train /= 255
        imgs_mask_train /= 255
        return imgs_train, imgs_guide_train, imgs_mask_train

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + self.custom_name + "_test.npy")
        imgs_guide_test = np.load(self.npy_path + self.custom_name + "_guide_test.npy")

        imgs_test = imgs_test.astype('float32')
        imgs_guide_test = imgs_guide_test.astype('float32')

        imgs_test /= 255
        imgs_guide_test /= 255

        return imgs_test, imgs_guide_test



if __name__ == "__main__":
    mydata = dataProcess(512, 512)
    mydata.create_train_data()
    mydata.create_test_data()
