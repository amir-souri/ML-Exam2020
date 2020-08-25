import cv2
from transformations import *
import random as rand
import os


img_laa = cv2.imread('inputs/laa_laa.jpg')
t, s, r = translating(21, 21), scaling(2, 2), rotating(45)
r2, t2, s2 = rotating(20), translating(7, 7), scaling(.5, .5)
com_tsr = combine(s, t, r)
com_rts = combine(r2, t2, s2)
tsr = transform_image(img_laa, com_tsr)
rts = transform_image(img_laa, com_rts)
cv2.imshow('Before', img_laa)
cv2.imshow('After_21translated_2scaled_45rotated', tsr)
cv2.imshow('After_20rotated_7translated_0.5scaled', rts)

if os.path.isdir("./outputs"):
    cv2.imwrite("./outputs/21t_2s_45r_laa.png", tsr)
    cv2.imwrite("./outputs/20r_7t_0.5s_laa.png", rts)
else:
    os.mkdir("./outputs")
    cv2.imwrite("./outputs/21t_2s_45r_laa.png", tsr)
    cv2.imwrite("./outputs/20r_7t_0.5s_laa.png", rts)

if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()


arb_x, arb_y = rand.sample(range(0, 501), 2)
arb_point = np.array([[arb_x],
                      [arb_y],
                      [1]])

p = np.array([[5, 5]])

manual_rts_trans = np.array([[0.46984631, 0.17101007, 3.5],
                             [-0.17101007, 0.46984631, 3.5],
                             [0, 0, 1]])


func = transform_points(com_rts, p)
man = make_euclidean((manual_rts_trans @ make_homogeneous(p).T).T)

np.testing.assert_array_almost_equal(x=func, y=man, decimal=2, err_msg='Fail')