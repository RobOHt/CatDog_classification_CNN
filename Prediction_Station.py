import cv2
import tensorflow as tf

Categories = ['Dog', 'Cat']

model = tf.keras.models.load_model("Cat_vs_Dog_CNN.model")


def getImg(PATH):
    size = 50
    unscaled_img_array = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    scaled_img_array = cv2.resize(unscaled_img_array, (size, size))
    return scaled_img_array.reshape(-1, size, size, 1)


while True:
    FILENAME = input('Input a file name please.')
    prediction = model.predict([getImg(FILENAME)])  # input is a list
    print("This is a {}".format(Categories[int(prediction[0, 0])]))
