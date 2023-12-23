"""
Different flavors of visual population receptive field models
==============================================================

In this example script we will try out increasingly complex models for
visual population receptive fields (PRFs). We will start with a simple
Gaussian PRF model, and then add more complexity step by step.

"""

# %%
# Load data
# ---------
# First we load in the data. We will use the Szinte (2024)-dataset.
from braincoder.utils.data import load_szinte2024

data = load_szinte2024()

# This is the visual stimulus ("design matrix")
paradigm = data['stimulus']
grid_coordinates = data['grid_coordinates']

# This is the fMRI response data
d = data['v1_timeseries']
tr = data['tr']


# %%
# Simple 2D Gaussian Recetive Field model
# -------------------------------------
# Now we set up a simple Gaussian PRF model
from braincoder.models import GaussianPRF2DWithHRF
from braincoder.hrf import SPMHRFModel
hrf_model = SPMHRFModel(tr=tr)
model = GaussianPRF2DWithHRF(data=d, paradigm=paradigm, hrf_model=hrf_model, grid_coordinates=grid_coordinates)

# %%
# And a parameter fitter...
from braincoder.optimize import ParameterFitter
par_fitter = ParameterFitter(model=model, data=d, paradigm=paradigm)


# %%
# Now we try out a relatively coarse grid search to find the some
# parameters to start the gradient descent from.
import numpy as np
x = np.linspace(-8, 8, 10)
y = np.linspace(-4, 4, 10)
sd = np.linspace(0.1, 4, 10)

# We start the grid search using a correlation cost, so ampltiude
# and baseline do not influence those results.
# We will optimize them later using OLS.
baseline = [0.0]
amplitude = [1.0]

# Let's double-check the order of the parameters
print(model.parameter_labels)

# Now we can do the grid search
pars_grid_search = par_fitter.fit_grid(x, y, sd, baseline, amplitude, correlation_cost=True)

pars_grid_search_ols = par_fitter.refine_baseline_and_amplitude(pars_grid_search)