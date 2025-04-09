from functools import partial
from webbrowser import Error

from scipy.interpolate import interp1d
from yaml import Loader
import os
import numpy as np
import json
import yaml

if os.getenv('DISABLE_JAX', False):
    import numpy as jnp
    from scipy.special import expit
    def jit(func):
        return func
else:
    import jax.numpy as jnp
    from jax.scipy.special import expit
    from jax import jit

class HEFTEmulator(object):

    """Main emulator object"""

    def __init__(
        self,
        forceLPT=True,
        training_file='training_data.json',
    ):
        """
        Initialize the emulator object. Default values for all kwargs were
        used for fiducial results in 2101.11014, so don't change these unless
        you have a good reason!

        Kwargs:
            forceLPT : bool
                Whether to transition to pure LPT at low k.
            training_file: str
                File that the training data is stored in.
        """
        self.nspec = 15
        self.forceLPT = forceLPT

        training_file_abspath = "/".join(
            [
                os.path.dirname(os.path.realpath(__file__)),
                "data",
                training_file,
            ]
        )

        with open(training_file_abspath, 'r') as f:
            fp = json.load(f)
            self.coeff = np.array(fp['pce_coefficients'])
            self.exp = np.array(fp['pce_exponents'])
            self.param_mean = np.array(fp['param_mean'])
            self.param_mult = np.array(fp['param_mult'])
            self.pcs_mean = np.array(fp['pcs_mean'])
            self.pcs_mult = np.array(fp['pcs_mult'])
            self.evec_spec = np.array(fp['evec_spec'])
            self.k = np.array(fp['k'])
            self.kmin = np.array(fp['k_min'])
            self.kmax = np.array(fp['k_max'])
            self.zs = np.array(fp['zs'])

        self.evec_spline = interp1d(
            self.k, self.evec_spec, axis=1, fill_value=0, bounds_error=False
        )

    def predict(self, k, cosmo, spec_lpt, k_lpt=None):
        """
        Make predictions from a trained emulator given a vector of wavenumbers and
        a cosmology.

        Args:
            k : array-like
                1d vector of wave-numbers. Maximum k cannot be larger than
                self.kmax. For k < self.kmin, predictions will be made using
                velocileptors, for self.kmin <= k < self.kmax predictions
                use the emulator.
            cosmo : array-like
                Vector containing cosmology/scale factor in the order
                (ombh2, omch2, w0, ns, 10^9 As, H0, mnu, sigma8(z)).
            spec_lpt : array-like
                LPT predictions for spectra from velocileptors at the specified cosmology
                call.
            k_lpt: array-like
                k values LPT predictions are evaluated at if different than k.

        Output:
            Emulator predictions for the basis spectra of the 2nd order lagrangian bias expansion.
            Since we are treating neutrinos, the lensing and clustering spectra trace the matter field ('1') and
            the cdm+baryon field ('cb') respectively. This means we have, in fact, 15 basis spectra.

            Order of spectra is 1-1 (ie the matter power spectrum), 1-cb, cb-cb, delta-1, delta-cb, delta-delta, delta2-1, delta2-cb, delta2-delta,
            delta2-delta2, s2-1, s2-cb, s2-delta, s2-delta2, s2-s2.
        """

        if len(cosmo.shape) == 1:
            x = cosmo[:, np.newaxis]
        else:
            # to keep API same as before for aemulus alpha
            x = cosmo.T

        if np.any(k > np.max(self.kmax)):
            if np.all(k > np.max(self.kmax)):
                raise (
                    ValueError(
                        "Trying to compute spectra beyond the maximum value of the emulator!"
                    )
                )
            else:
                print(
                    "{} is greater than k_max for at least one spectrum. Setting P(k>k_max)=0".format(
                        np.max(k)
                    )
                )

        # scale input variables
        x_n = (x - self.param_mean[:, np.newaxis]) * self.param_mult[:, np.newaxis]

        in_domain = (-1.0001 <= x_n) & (x_n <= 1.0001)

        #allow for zero neutrino mass extrapolation
        if not (in_domain.all(axis=0) | ((x[6,:]>=0) & (x_n[6,:] <= 1.0001))).all():
            raise (ValueError("{} is not in training domain".format(x[~in_domain])))

        # evaluate lpt spectra at correct k if not already
        if (k_lpt is not None) & (np.sum(k != k_lpt) > 0):
            lpt_interp = interp1d(k_lpt, spec_lpt, axis=-1, fill_value="extrapolate")
            spectra_lpt = lpt_interp(k)
        else:
            spectra_lpt = spec_lpt

        if spectra_lpt.shape[-1] != len(k):
            raise (
                ValueError(
                    "Trying to feed in lpt spectra computed at different k than the desired outcome!"
                )
            )

        # interpolate PCs
        evecs = self.evec_spline(k)

        lambda_surr_normed = np.sum(
            self.coeff[..., np.newaxis]
            * np.prod(x_n ** self.exp[..., np.newaxis], axis=-2)[:, np.newaxis, :],
            axis=-2,
        )
        lambda_surr = (
            lambda_surr_normed / self.pcs_mult[..., np.newaxis]
            + self.pcs_mean[..., np.newaxis]
        )
        simoverlpt_emu = np.einsum("bkp, bpc->cbk", evecs, lambda_surr)

        pk_emu = np.zeros_like(spectra_lpt)
        pk_emu[:] = spectra_lpt
        # set spectra above kmax to 0
        pk_emu[..., k[np.newaxis, :] > self.kmax[:, np.newaxis]] = 0

        # Enforce agreement with LPT
        if self.forceLPT:
            if len(self.kmin.shape)>1:
                assert(self.kmin.shape[1]==self.nspec)

                for i in range(self.nspec):
                    pk_emu[:,i,k > self.kmin[:, i]] = (
                        10 ** (simoverlpt_emu) * pk_emu
                        )[..., k > self.kmin[:, i]]
            else:
                pk_emu[..., k[np.newaxis, :] > self.kmin[:, np.newaxis]] = (
                    10 ** (simoverlpt_emu) * pk_emu
                )[..., k[np.newaxis, :] > self.kmin[:, np.newaxis]]
        else:
            pk_emu[...] = 10 ** (simoverlpt_emu) * pk_emu[...]

        return pk_emu

    def error_covariance(self, spec_heft, k, z, frac_error_cov):
        """Compute the emulator error covariance for a given set
        of HEFT spectra, provided the k and z values used to compute those spectra.

        Args:
            spec_heft array-like: (15,Nk) array containing HEFT spectra
            k array-like: (Nk) array containing k values that spec_heft is evaluated at.
            z float: redshift value that spec_heft is evaluated at.
            frac_error_cov array-like: (30,15,Nkp,Nkp) array containing the aemulus nu fractional error covariance.

        Returns:
            cov array-like: (15,Nk,Nk) array containing the aemulus nu emulator error covariance for the provided spectra.
        """

        cov = interp1d(self.zs, frac_error_cov, axis=0, kind='cubic')(z)
        cov = interp1d(self.k, cov, axis=1, bounds_error=False, fill_value=0, kind='cubic')(k)
        cov = interp1d(self.k, cov, axis=2, bounds_error=False, fill_value=0, kind='cubic')(k)
        cov *= spec_heft[:,:,np.newaxis] * spec_heft[:,np.newaxis,:]

        return cov


    def basis_to_full(self, k, btheta, emu_spec, cross=True):
        """
        Take an LPTemulator.predict() array and combine with bias parameters to obtain predictions for P_hh and P_hm.


        Inputs:
        -k: set of wavenumbers used to generate emu_spec.
        -btheta: vector of bias + shot noise. See notes below for structure of terms
        -emu_spec: output of LPTemu.predict() at a cosmology / set of k values
        -halomatter: whether we compute only P_hh or also P_hm

        Outputs:
        -pfull: P_hh (k) or a flattened [P_hh (k),P_hm (k)] for given spectrum + bias params.


        Notes:
        Bias parameters can either be

        btheta = [b1, b2, bs2, SN]

        or

        btheta = [b1, b2, bs2, bnabla2, SN]

        Where SN is a constant term, and the bnabla2 terms follow the approximation

        <X, nabla^2 delta> ~ -k^2 <X, 1>.

        Note the term <nabla^2, nabla^2> isn't included in the prediction since it's degenerate with even higher deriv
        terms such as <nabla^4, 1> which in principle have different parameters.

        """
        if len(btheta) == 4:
            b1, b2, bs, sn = btheta
            # Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
            bterms_hh = [
                0,
                0,
                1,
                0,
                2 * b1,
                b1**2,
                0,
                b2,
                b2 * b1,
                0.25 * b2**2,
                0,
                2 * bs,
                2 * bs * b1,
                bs * b2,
                bs**2,
            ]

            # hm correlations only have one kind of <1,delta_i> correlation
            bterms_hm = [0, 1, 0, b1, 0, 0, b2 / 2, 0, 0, 0, bs, 0, 0, 0, 0]

            pkvec = emu_spec

        else:
            b1, b2, bs, bk2, sn = btheta
            # Cross-component-spectra are multiplied by 2, b_2 is 2x larger than in velocileptors
            bterms_hh = [
                0,
                0,
                1,
                0,
                2 * b1,
                b1**2,
                0,
                b2,
                b2 * b1,
                0.25 * b2**2,
                0,
                2 * bs,
                2 * bs * b1,
                bs * b2,
                bs**2,
                #the bnabla spectra start here
                2 * bk2,
                2 * bk2 * b1,
                bk2 * b2,
                2 * bk2 * bs,
            ]

            # hm correlations only have one kind of <1,delta_i> correlation
            bterms_hm = [
                0,
                1,
                0,
                b1,
                0,
                0,
                b2 / 2,
                0,
                0,
                0,
                bs,
                0,
                0,
                0,
                0,
                bk2,
                0,
                0,
                0,
            ]

            pkvec = np.zeros(shape=(self.nspec + 4, len(k)))
            pkvec[: self.nspec] = emu_spec

            # IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.
            if cross:
                nabla_idx = [1, 3, 6, 10]
            else:
                nabla_idx = [2, 4, 7, 11]

            # Higher derivative terms
            pkvec[self.nspec :] = -(k**2) * pkvec[nabla_idx]

        if cross:
            bterms_hm = jnp.array(bterms_hm)
            pfull = jnp.einsum("b, bk->k", bterms_hm, pkvec)

        else:
            bterms_hh = jnp.array(bterms_hh)
            pfull = jnp.einsum("b, bk->k", bterms_hh, pkvec) + sn

        return pfull


param_indices = {'As': 0, 'ns': 1, 'H0': 2, 'w0': 3, 'ombh2': 4, 'omch2': 5,'mnu': 6, 'z': 7}
class NNScalarEmulator(object):
    def __init__(self, filebase, rescale_weights={'As': 1e10}):
        super(NNScalarEmulator, self).__init__()

        self.load(filebase, rescale_weights=rescale_weights)

    @property
    def n_parameters(self):
        return self.layer_sizes[0]

    @property
    def n_components(self):
        return self.layer_sizes[-1]

    @property
    def n_layers(self):
        return len(self.W)

    def load(self, filebase, rescale_weights=None):
        with open('{}.json'.format(filebase), 'r') as fp:
            weights = json.load(fp)
            self.layer_sizes = [len(weights['W'][i]) for i in range(len(weights['W']))]+[len(weights['W'][-1][0])]
            max_layer_size = max(self.layer_sizes)
            for k in weights:
                if k in ['W']:
                    tmp_weights = jnp.empty((len(weights[k]), max_layer_size, max_layer_size), dtype=jnp.float32)
                    for i, wi in enumerate(weights[k]):
                        tmp_weights = tmp_weights.at[i, :self.layer_sizes[i], :self.layer_sizes[i+1]].set(wi)
                    weights[k] = tmp_weights
                elif k in ['b', 'alphas', 'betas']:
                    tmp_weights = jnp.empty((len(weights[k]), max_layer_size), dtype=jnp.float32)
                    for i, wi in enumerate(weights[k]):
                        tmp_weights = tmp_weights.at[i, :self.layer_sizes[i + 1]].set(wi)
                    weights[k] = tmp_weights
                elif k in ["param_mean", "param_sigmas"]:
                    weights[k] = jnp.array(weights[k]).astype(jnp.float32)
                    if rescale_weights is not None:
                        for wk in rescale_weights:
                            weights[k] = weights[k].at[param_indices[wk]].multiply(rescale_weights[wk])
                else:
                    weights[k] = jnp.array(weights[k]).astype(jnp.float32)

                setattr(self, k, weights[k])


    @staticmethod
    @jit
    def activation(x, alpha, beta):
        return (beta + (expit(alpha * x) * (1 - beta))) * x

    @jit
    def __call__(self, parameters):

        outputs = []
        x = (parameters - self.param_mean) / self.param_sigmas

        for i in range(self.n_layers - 1):

            # linear network operation
            x = x @ self.W[i, :self.layer_sizes[i], :self.layer_sizes[i+1]] + self.b[i, :self.layer_sizes[i+1]]

            # non-linear activation function
            x = self.activation(x, self.alphas[i, :self.layer_sizes[i+1]], self.betas[i, :self.layer_sizes[i+1]])

        # linear output layer
        x = (jnp.sum(x[...,None] * self.W[-1, :self.layer_sizes[-2], :self.layer_sizes[-1]][None,...], axis=1) + self.b[-1, :self.layer_sizes[-1]]) * self.pc_sigmas + self.pc_mean

        return x


class NNSpecEmulator(NNScalarEmulator):
    def __init__(self, filebase, kmin=1e-3, kmax=0.5, rescale_weights={'As': 1e10}):
        super(NNSpecEmulator, self).__init__(filebase, rescale_weights=rescale_weights)

        self.nk = self.sigmas.shape[0]
        self.k = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), self.nk)

    @jit
    def __call__(self, parameters):

        outputs = []
        x = (parameters - self.param_mean) / self.param_sigmas

        for i in range(self.n_layers - 1):
            # linear network operation
            x = x @ self.W[i, :self.layer_sizes[i], :self.layer_sizes[i + 1]] + self.b[i, :self.layer_sizes[i + 1]]

            # non-linear activation function
            x = self.activation(x, self.alphas[i, :self.layer_sizes[i + 1]], self.betas[i, :self.layer_sizes[i + 1]])

        # linear output layer
        x = (jnp.sum(x[...,None] * self.W[-1, :self.layer_sizes[-2], :self.layer_sizes[-1]][None,...], axis=1) + self.b[-1, :self.layer_sizes[-1]]) * self.pc_sigmas[:self.n_components] + self.pc_mean[:self.n_components]

        x = jnp.sinh((x @ self.v[:, :self.n_components].T)
                    * self.sigmas + self.mean) * self.fstd

        return self.k, x

class NNHEFTEmulator(HEFTEmulator):

    def __init__(self, config='nn_cfg.yaml', abspath=False, rescale_weights={'As': 1e10}, sigma8z_emu=None):

        if not abspath:
            data_dir =  "/".join(
                [
                    os.path.dirname(os.path.realpath(__file__)),
                    "data",
                ]
            )

            if type(config) is not dict:
                config_abspath = "/".join(
                    [
                        os.path.dirname(os.path.realpath(__file__)),
                        "data",
                        config,
                    ]
                )
                with open(config_abspath, 'r') as fp:
                    cfg = yaml.load(fp, Loader=Loader)
            else:
                cfg = config
        else:
            if type(config) is not dict:
                config_abspath = config
                with open(config_abspath, 'r') as fp:
                    cfg = yaml.load(fp, Loader=Loader)
            else:
                cfg = config

        pij_emu_bases = cfg['pij_bases']
        s8z_base = cfg['s8z_base']
        kmin = float(cfg['kmin'])
        kmax = float(cfg['kmax'])

        assert(len(pij_emu_bases) == self.nspec)

        self.pij_emus = []

        for i in range(self.nspec):
            if not abspath:
                self.pij_emus.append(NNSpecEmulator(f'{data_dir}/{pij_emu_bases[i]}', kmin=kmin, kmax=kmax, rescale_weights=rescale_weights))
            else:
                self.pij_emus.append(NNScalarEmulator(pij_emu_bases[i], kmin=kmin, kmax=kmax, rescale_weights=rescale_weights))

        if sigma8z_emu is None:
            if not abspath:
                self.sigma8z_emu = NNScalarEmulator(f'{data_dir}/{s8z_base}', rescale_weights=rescale_weights)
            else:
                self.sigma8z_emu = NNScalarEmulator(s8z_base, rescale_weights=rescale_weights)
        else:
            self.sigma8z_emu = sigma8z_emu

    @property
    def nspec(self):
        return 15

    @property
    def param_order(self):
        return [4, 3, 5, 2, 0, 1, 6, 7]

    @jit
    def predict(self, parameters):
        """
        Make predictions from a trained neural network emulator for a given cosmology and redshift.
        Args:
            parameters : array-like (Npred, 8)
                Vector containing cosmology/redshift in the order
                (ombh2, omch2, w0, ns, 10^9 As, H0, mnu, z).

        Output:
            k : array-like (Nk)
              Wavenumbers corresponding to the emulator prediction in units of Mpc/h.
            pij: array-like (Npred, 15, Nk)
                Emulator predictions for the basis spectra of the 2nd order lagrangian bias expansion.
                Since we are treating neutrinos, the lensing and clustering spectra trace the matter field ('1') and
                the cdm+baryon field ('cb') respectively. This means we have, in fact, 15 basis spectra.

                Order of spectra is 1-1 (ie the matter power spectrum), 1-cb, cb-cb, delta-1, delta-cb, delta-delta, delta2-1, delta2-cb, delta2-delta,
                delta2-delta2, s2-1, s2-cb, s2-delta, s2-delta2, s2-s2.
        """
        parameters = jnp.atleast_2d(parameters)
        params = parameters[:, self.param_order]
        params = params.at[:, 0].set(jnp.exp(params[:, 0]))
        params = params.at[:, -2].set(jnp.log10(params[:, -2]))

        s8z = self.sigma8z_emu(params)[:, 0]
        params = params.at[:, -1].set(s8z)
        npred = len(params)

        k = self.pij_emus[0].k

        pij = jnp.zeros((npred, self.nspec, len(k)))

        for i in range(self.nspec):
            pij = pij.at[:, i, :].set(self.pij_emus[i](params)[1])

        return k, pij


if not os.getenv('DISABLE_JAX', False):
    from jax._src.tree_util import register_pytree_node

    class jax_emu_aux_functions:
        all_emu_attr = ['W', 'b', 'alphas', 'betas', 'param_mean', 'param_sigmas', 'pc_mean', 'pc_sigmas']
        spec_emu_extra_attr = ['v', 'sigmas', 'mean', 'fstd', 'k']
        all_emu_aux_attr = ['layer_sizes', 'nk']

        @staticmethod
        def _flatten__NNScalarEmulator(obj):
            attr = [getattr(obj, k, None) for k in jax_emu_aux_functions.all_emu_attr]
            aux = [getattr(obj, k, None) for k in jax_emu_aux_functions.all_emu_aux_attr]
            return tuple(attr), tuple(aux)

        @staticmethod
        def _flatten__NNSpecEmulator(obj):
            attr = [getattr(obj, k, None) for k in jax_emu_aux_functions.all_emu_attr+jax_emu_aux_functions.spec_emu_extra_attr]
            aux = [getattr(obj, k, None) for k in jax_emu_aux_functions.all_emu_aux_attr]
            return tuple(attr), tuple(aux)

        @staticmethod
        def _flatten_NNHEFTEmulator(obj):
            return (obj.pij_emus, obj.sigma8z_emu), ()

        @staticmethod
        def _reconstruct__NNScalarEmulator(aux_data, children):
            emu = object.__new__(NNScalarEmulator)
            for i,k in enumerate(jax_emu_aux_functions.all_emu_attr):
                setattr(emu, k, children[i])
            for i,k in enumerate(jax_emu_aux_functions.all_emu_aux_attr):
                setattr(emu, k, aux_data[i])

            return emu

        @staticmethod
        def _reconstruct__NNSpecEmulator(aux_data, children):
            emu = object.__new__(NNSpecEmulator)
            for i, k in enumerate(jax_emu_aux_functions.all_emu_attr+jax_emu_aux_functions.spec_emu_extra_attr):
                setattr(emu, k, children[i])
            for i, k in enumerate(jax_emu_aux_functions.all_emu_aux_attr):
                setattr(emu, k, aux_data[i])

            return emu

        @staticmethod
        def _reconstruct__NNHEFTEmulator(aux_data, children):
            emu = object.__new__(NNHEFTEmulator)
            emu.pij_emus = children[0]
            emu.sigma8z_emu = children[1]
            return emu

    register_pytree_node(NNScalarEmulator, jax_emu_aux_functions._flatten__NNScalarEmulator, jax_emu_aux_functions._reconstruct__NNScalarEmulator)
    register_pytree_node(NNSpecEmulator, jax_emu_aux_functions._flatten__NNSpecEmulator, jax_emu_aux_functions._reconstruct__NNSpecEmulator)
    register_pytree_node(NNHEFTEmulator, jax_emu_aux_functions._flatten_NNHEFTEmulator, jax_emu_aux_functions._reconstruct__NNHEFTEmulator)