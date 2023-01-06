# Example of Custom Transformers

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

rooms_ix, bedrooms_ix, population_ix, housolds_ix = 3, 4, 5, 6

# we inherit from some base class
# we need to provide a couple of methods:
# 1) fit, just returns itself
# 2) transform
# 3) fit_transform() => if you inherit from TransformerMixin, you get it from free (usually is just call fit() and then transform())

# inherit from base class for estimator
# inherit

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_poer_room = True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_poer_room

    def fit(self, X, y = None):
        return self #nothing else to do

    def transform(self, X):
        rooms_per_housold = X[:, rooms_ix] / X[:, housolds_ix]
        population_per_housold = X[:, population_ix] / X[:, housolds_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_housold, population_per_housold, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_housold, population_per_housold]

#to use it in the exercise 

# attr_adder = CombinedAttributesAdder(add_bedrooms_poer_room=False)
# housing_extra_attribs = attr_adder.transform(housing.values)

#Transformation pipelines!

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#let's build a pipeleine for our numerical columns

num_pipeline = Pipeline([('imputer'), SimpleImputer(strategy="median"), ('attribs_adder', CombinedAttributesAdder()),('std_scaler', StandardScaler()),])