import copy
import math
from sklearn import metrics
from utils import *
from detector import *
import matplotlib.pyplot as plt
import sys

if len(sys.argv) == 1:
    raise ValueError("You must supply data folder as a program argument.")

dataset = sys.argv[1]


def get_my_pup_cen(dataset):
    imgs = load_images(dataset)
    my_pupils = [find_pupil(imgs[i], debug=False) for i in range(len(imgs))]
    my_pupils_centers = np.zeros((len(my_pupils), 2))
    for i in range(len(my_pupils)):
        my_pupils_centers[i] = my_pupils[i][0]

    return my_pupils_centers


def get_pup_dis_error(dataset):
    ground_pupils = load_json(dataset, 'pupils')
    cx = np.zeros((len(ground_pupils), 1))
    cy = np.zeros((len(ground_pupils), 1))
    for i in range(len(ground_pupils)):
        cx[i] = ground_pupils[i]['cx']
        cy[i] = ground_pupils[i]['cy']

    ground_pupils_centers = np.hstack((cx, cy))
    imgs = load_images(dataset)
    my_pupils_centers = get_my_pup_cen(dataset)
    pup_dis_error = np.zeros(len(my_pupils_centers))
    for i in range(len(imgs)):
        pup_dis_error[i] = dist(np.asarray(ground_pupils_centers[i]), np.asarray(my_pupils_centers[i]))

    rms = np.sqrt(metrics.mean_squared_error(my_pupils_centers, ground_pupils_centers))
    return pup_dis_error, rms

def get_gli_dis_error(dataset):
    imgs = load_images(dataset)
    ground_glints = load_json(dataset, 'glints')
    my_pupils_centers = get_my_pup_cen(dataset)
    my_glints = [find_glints(imgs[i], my_pupils_centers[i], debug=False) for i in range(len(imgs))]
    dis_err = copy.deepcopy(my_glints)
    em = []
    for i in range(len(imgs)):
        for m in range(len(my_glints[i])):
            for g in range(len(ground_glints[i])):
                em.append(dist(np.asarray(my_glints[i][m]), np.asarray(ground_glints[i][g])))
                dis_err[i][m] = min(em)
            em.clear()

    return dis_err


if __name__ == "__main__":
    pup_dis_error, rms = get_pup_dis_error(dataset)
    pup_mean = pup_dis_error.mean()
    pup_median = np.median(pup_dis_error)
    print(f"mean of distances error for pupils ({dataset})", pup_mean)
    print(f"median of distances error for pupils ({dataset})", pup_median)
    print(f'Root Mean Squared Error for pupils ({dataset}):', rms)

    n_p = math.ceil(max(pup_dis_error))+2
    binsp = np.arange(0, n_p, 0.5)
    plt.hist(pup_dis_error, bins=binsp, align = 'left', edgecolor = 'w')
    plt.xlabel('Distance error (euclidean)', fontsize=21)
    plt.ylabel('Number of pupils', fontsize=21)
    plt.xticks(binsp)
    plt.annotate(f'n={len(pup_dis_error)}', xy=(0.80, 0.90), xycoords='axes fraction', size=21)
    plt.gcf().savefig(f'outputs/pup_dis_error_his_{dataset}.png')
    plt.show()

    flatten = lambda l: [item for sublist in l for item in sublist]
    gli_dis_error = get_gli_dis_error(dataset)
    gli_dis_error_flat = flatten(gli_dis_error)
    gli_dis_error_flat = np.array(gli_dis_error_flat)
    gli_mean = gli_dis_error_flat.mean()
    gli_median = np.median(gli_dis_error_flat)
    n_g = math.ceil(max(gli_dis_error_flat)) + 2
    binsg = np.arange(0, n_g, 0.5)
    plt.hist(gli_dis_error_flat, bins=binsg,  align = 'left', edgecolor = 'w')
    plt.xlabel('Distance error (euclidean)', fontsize=21)
    plt.ylabel('Number of glints', fontsize=21)
    plt.xticks(np.arange(0, n_g, 2))
    plt.annotate(f'n={len(gli_dis_error_flat)}', xy=(0.80, 0.90), xycoords='axes fraction', size=21)
    plt.gcf().savefig(f'outputs/gli_dis_error_his_{dataset}.png')
    plt.show()

    print(f"mean of distances error for glints ({dataset})", gli_mean)
    print(f"median of distances error for glints ({dataset})", gli_median)
