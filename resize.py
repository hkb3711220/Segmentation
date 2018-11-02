import cv2
import os

os.chdir(os.path.dirname(__file__))
PATH = os.path.dirname(__file__)
SAVE_PATH = os.path.join(PATH, 'Resized')
ori_imgs_name = os.listdir('./Pictures')

for img_name in ori_imgs_name:
    img = cv2.imread('./Pictures/{}'.format(img_name), cv2.IMREAD_COLOR)
    size = (227, 227)
    resized_img = cv2.resize(img, size)
    save_img_name = os.path.join(SAVE_PATH, img_name)

    cv2.imwrite(save_img_name, resized_img)
