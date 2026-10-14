.. meta::
   :description: What NS-SDN does that a reader of the papers may not expect: the trend head absorbing the series, the weekly-fill frequency artifact, permutation identifiability, criteria caveats, short-schedule behaviour, and what reproduces across machines.

Nuances
=======

Things the model does that are correct, documented, and still surprising
on first contact. Each entry says what happens, why, and what to do
about it.

The trend head absorbs the series
---------------------------------

A GRU state with a linear trend head can represent the current level of
the series almost perfectly, and under the filtered alignment the
training target *is* the current observation. The cheapest way for the
network to lower the loss early in training is therefore to route the
series through :math:`B_t` and leave the sinusoids small, and on short
schedules that is what it does: ``trend`` tracks the series, the
amplitudes are modest, and the frequencies settle wherever the
initialization and the penalties leave them rather than at the
oscillations in the data. The quickstart shows it on a series with two
known sinusoids.

Under the predicted alignment the same mechanism becomes persistence:
the trend head reproduces :math:`y_{t-1}` and the one-step error is the
one-step change. On a synthetic chirp the package's own tests found
essentially none of the oscillatory variance in the components under
that alignment, and the ``trend`` penalty does not change the outcome
on its own.

The published schedule is long (three thousand steps) and penalized
precisely so that the amplitude and frequency paths become smooth,
readable objects on a series with slow spectral structure; on the
DGS10 run the components do carry structure, which is what the
IEEE-ORSS figures show. Read any fit's ``power_share`` and
``oscillation`` against ``trend`` before reading its components, and
treat the frequency paths of a short exploratory fit as initialization
plus drift, not as a spectrum.

The weekly-fill artifact
------------------------

The published DGS10 series is placed on a calendar-day grid and gaps of
up to seven days are filled forward, so every weekend is two repeated
values. That is a seven-day structure in the input, and
:math:`2\pi / 7 \approx 0.898` radians per day is close to the lowest
frequency the published run reports (0.90 to 0.95). The softplus head
initializes near :math:`\log 2 \approx 0.693`, so a component that
drifts up to the weekly frequency is doing cheap work on the fill rather
than finding a cycle in yields.

Keep the published grid when reproducing the papers; for new analysis,
fit on the business-day series with the integer grid and note that the
frequency unit is then radians per trading day. Compare the frequency
paths of the two fits before attributing a component to the economy.

Components are identified up to permutation
-------------------------------------------

Nothing in the default model orders the components, so the index
:math:`k` is arbitrary: refit with another seed and component 3 may be
component 11. The summary's last note says so. Comparisons across fits
must therefore go through permutation-invariant objects, which is what
the :doc:`spectral state <spectral_state>` is for, or through
``ordered_frequency=True``, which parameterizes the frequencies as
increasing and identifies the components by rank. The ordered
parameterization is a different model from the published one (it
changes the frequency head and the parameter count, and it confines the
frequencies to the Nyquist band), so say which you used.

Compression is where disentanglement is weakest
-----------------------------------------------

Two components with the same frequency and offset are one component
with a different amplitude; as frequencies converge, the split of power
between them is not determined by the data. The spectral-compression
reading of 2020 is therefore exactly the regime in which per-component
amplitudes and gates are least identified. ``disentanglement_score``
and ``active`` are the checks, and the measure-level coordinates
(centroid, dispersion, effective count) are the quantities that remain
meaningful through it.

Two RMSEs, one fit
------------------

Under the filtered alignment the summary reports an *Emission RMSE*
and a *One-step RMSE*, and they can differ by an order of magnitude.
The first is the error of reconstructing :math:`y_t` from a state that
has seen :math:`y_t`; the second is the error of carrying that emission
forward as the forecast of :math:`y_{t+1}`. Only the second is a
forecast error, only the second enters ``llf`` and the criteria, and
even it should be read against the random walk. See
:doc:`forecasting`.

Information criteria charge the nominal parameter count
-------------------------------------------------------

``n_params`` is the number of trainable weights, which for a weight-
decayed, drift-penalized network overstates its effective degrees of
freedom, and ``llf`` is a Gaussian likelihood concentrated at a
penalized optimum rather than a maximum. BIC in particular will prefer
the smallest ``k`` almost regardless of fit. Use the criteria to rank
values of ``k`` within one engine configuration and seed, and prefer
the holdout one-step error and the spectral-state coordinates for
decisions that matter.

Short schedules are exploratory
-------------------------------

Every example in this documentation fits in seconds with a one-stage
schedule of tens of steps. That is enough to exercise the API and to
see the shapes and identities, and not enough for the components to
mean anything. Fits reported in a paper should use the published
schedule or a documented alternative, with ``seed`` set and a seed
sweep for any claim about a particular component.

What reproduces across machines
-------------------------------

The architecture and its parameter count; the validation, the scaling
and the alignment logic; the identities between the paths. Trained
weights reproduce exactly on one machine with one seed, and
approximately across BLAS backends and torch versions. The published
58,849-parameter count is checked by the test suite; the published
metrics are reproduced by the :doc:`reproduction example <../auto_examples/plot_03_reproduce_orss>` to the tolerance those differences
allow.

Torch threads
-------------

Training is a sequence of small recurrent passes, which gains little
from many threads and can lose to contention on shared CI runners.
``torch.set_num_threads`` before fitting is the control; two to four
threads is usually the fastest setting for the published size.
