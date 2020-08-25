from sklearn import metrics
from utils import *
from detector import *
import matplotlib.pyplot as plt
import sys
from test_detector import get_pup_dis_error
import gaze

if len(sys.argv) == 1:
    raise ValueError("You must supply data folder as a program argument.")

dataset = sys.argv[1]


def get_euclidean_distance_gazes(dataset):
    global model
    imgs = load_images(dataset)
    ground_pos = np.array(load_json(dataset, 'positions'))
    # model = gaze.GazeModel(imgs[:9], ground_pos[:9])
    model = gaze.PolynomialGaze(imgs[:9], ground_pos[:9], order=2)
    imgs_not_used = imgs[9:]
    my_gaze_pos = [model.estimate(imgs_not_used[i]) for i in range(len(imgs_not_used))]
    my_gaze_pos = np.asarray([[np.round(i) for i in nested] for nested in my_gaze_pos])
    ground_pos_not_used = ground_pos[9:]
    euclidean_distance_gazes = np.zeros(len(my_gaze_pos))
    for i in range(len(my_gaze_pos)):
        euclidean_distance_gazes[i] = dist(my_gaze_pos[i], ground_pos_not_used[i])

    rms = np.sqrt(metrics.mean_squared_error(my_gaze_pos, ground_pos_not_used))

    return euclidean_distance_gazes, rms


gd, rms = get_euclidean_distance_gazes(dataset)
gaze_mean = gd.mean()
gaze_median = np.median(gd)
print(f"mean of distances error for gazes ({dataset})", gaze_mean)
print(f"median of distances error for gazes ({dataset})", gaze_median)
print(f'Root Mean Squared Error for gazes ({dataset}):', rms)

if isinstance(model, gaze.GazeModel):
    m = "Linear"
else:
    m = "Polynomial"

plt.hist(gd, cumulative=True, density=True)
plt.xlabel('Distance error (euclidean)', fontsize=21)
plt.ylabel(f'fractions of occurrences ({m})', fontsize=18)
plt.gcf().savefig(f'outputs/gaz_dis_error_his_{dataset}.png')
plt.show()

pd, _ = get_pup_dis_error(dataset)
co = np.corrcoef(gd, pd[9:])
print("Correlation between gaze distance errors and pupil distance error: \n", co)
