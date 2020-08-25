import transformations as trans
import random as rand
import numpy as np

for test in range(100):
    arb = rand.sample(range(0, 501), 6)
    sourcePoints = np.array([[arb[0], arb[1]], [arb[2], arb[3]], [arb[4], arb[5]]])
    homo_sourcePoints = trans.make_homogeneous(sourcePoints)
    sx, sy = rand.sample(range(-10, 11), 2)
    s = trans.scaling(sx, sy)
    tx, ty = rand.sample(range(-400, 401), 2)
    t = trans.translating(tx, ty)
    θ = rand.randint(-360, 361)
    r = trans.rotating(θ)
    compound = trans.combine(s, t, r)
    targetPoint = compound @ homo_sourcePoints.T
    euc_targetPoint = trans.make_euclidean(targetPoint.T)
    aff = trans.learn_affine(sourcePoints, euc_targetPoint)
    np.testing.assert_array_almost_equal(x=compound, y=aff, decimal=4,
                                         err_msg='The inferred affine transformation using the function learn_affine() '
                                                 'is not equal to the resulting transformation matrix with the one you '
                                                 'created.')
print("100 test passed")
