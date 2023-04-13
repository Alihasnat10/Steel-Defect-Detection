import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import dice_coef, mask2contour, mask_padding

def load_and_prep_image(filename, img_shape=(128,800)):
    img = cv2.imread(filename)
    img = cv2.resize(img, (800, 128))
    img = img/255.
    return img

def pred_and_plot(model, filename, class_names):
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0), verbose=0)
    pred_class = class_names[int(tf.round(pred)[0][0])]
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
    plt.show()
    return np.argmax(pred)

def load_models(classify_model_pth="models/defect_classify", segment_model_pth="models/segmentation/segment_model.h5"):
    classify_model = tf.keras.models.load_model(classify_model_pth)
    segment_model = tf.keras.models.load_model(segment_model_pth, custom_objects={"dice_coef":dice_coef})
    return classify_model, segment_model

def classify_inference(model, img_path="data/train_images/0a3bbea4d.jpg"):
    class_names = ["1", "2", "3", "4"]

    output_class = pred_and_plot(model=model, 
                filename=img_path, 
                class_names=class_names)
    return output_class

def segment_inference(segment_model, img_path="data/train_images/0a3bbea4d.jpg"):
    img = cv2.imread(img_path)
    img = img.copy()
    img = cv2.resize(img, (800, 128))
    res=segment_model.predict(np.expand_dims(img,axis=0))
    res_bool = (res>0.5).astype(np.uint8)[0]
    defect = []
    for j in range(4):
        msk = res_bool[:,..., j]
        msk = mask_padding(msk)
        msk = mask2contour(msk)
        if np.sum(msk) != 0:
            defect.append(j+1)
        if j == 0:
            img[msk==1] = 1, 0, 0
        elif j == 1:
            img[msk==1] = 0, 1, 0
        elif j == 2:
            img[msk==1] = 1, 1, 0
        elif j == 3:
            img[msk==1] = 1, 0, 1
            
    plt.title(f'defect: {defect}')
    plt.axis('off')
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    classify_model, segment_model = load_models()
    image_to_predict = "data/train_images/ffe93442c.jpg"
    classify_inference(classify_model, img_path=image_to_predict)

    segment_inference(segment_model, img_path=image_to_predict)