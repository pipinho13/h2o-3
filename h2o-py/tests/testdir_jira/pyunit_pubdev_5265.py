import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator

from tests import pyunit_utils


def pubdev_5265():
    # The MeanImputation strategy actually counts 'mode' in case of GLM.
    # There are more Bs, so the first 'nan' value should be replaced by '2'
    data_missing_A = {
        'first': ['A', 'A', 'A', 'A', 'A',
                  'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C'],
        'second': ['nan', 1, 1, 1, 1,
                   2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
    }

    data_missing_A_test = {
        'first': ['A', 'A', 'A', 'A', 'A',
                  'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C'],
        'second': ['nan', 1, 1, 1, 1,
                   2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
    }

    data_missing_A = h2o.H2OFrame(data_missing_A)
    data_missing_A['second'] = data_missing_A['second'].asfactor()

    data_missing_A_test = h2o.H2OFrame(data_missing_A_test)
    data_missing_A_test['second'] = data_missing_A_test['second'].asfactor()

    a_estimator = H2OGeneralizedLinearEstimator(family="multinomial", missing_values_handling="MeanImputation",
                                                seed=1234, Lambda=0, )
    a_estimator.train(x=["second"], y="first", training_frame=data_missing_A)
    #a_estimator.model_performance(test_data=data_missing_A_test)
    # 11 rows in total, 4 with A, 7 with B (6 in original data)

if __name__ == "__main__":
    pyunit_utils.standalone_test(pubdev_5265)
else:
    pubdev_5265()
