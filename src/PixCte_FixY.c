#include <stdlib.h>
#include <stdio.h>

#include <Python.h>
#include <numpy/arrayobject.h>

#include "PixCteCorr.h"

static PyObject * py_CalcCteFrac(PyObject *self, PyObject *args) {
  /* input variables */
  const double mjd;
  PyObject *opy_scale_mjd, *opy_scale_val;
  PyArrayObject *py_scale_mjd, *py_scale_val;
  
  /* local variables */
  double cte_frac;
  
  /* put arguments into variables */
  if (!PyArg_ParseTuple(args, "dOO", &mjd, &opy_scale_mjd, &opy_scale_val)) {
    return NULL;
  }
  
  py_scale_mjd = (PyArrayObject *) PyArray_FROMANY(opy_scale_mjd, NPY_DOUBLE, 
                                                    1, 1, NPY_IN_ARRAY);
  py_scale_val = (PyArrayObject *) PyArray_FROMANY(opy_scale_val, NPY_DOUBLE, 
                                                    1, 1, NPY_IN_ARRAY);
  
  /* get cte_frac */
  cte_frac = CalcCteFrac(mjd, (double *) PyArray_DATA(py_scale_mjd),
                         (double *) PyArray_DATA(py_scale_val));
  
  /* test whether it's good */
  if (cte_frac < 0) {
    PyErr_SetString(PyExc_ValueError,"No suitable CTE scaling data found in PCTETAB");
    return NULL;
  }
  
  Py_DECREF(py_scale_mjd);
  Py_DECREF(py_scale_val);
  
  return Py_BuildValue("d",cte_frac);
}


static PyObject * py_InterpolatePsi(PyObject *self, PyObject *args) {
  /* input variables */
  PyObject *opy_chg_leak, *opy_psi_node;
  PyArrayObject *py_chg_leak, *py_psi_node;
  
  /* local variables */
  int status;
  
  /* return variable */
  npy_intp chg_leak_dim[] = {MAX_TAIL_LEN, NUM_LOGQ};
  PyArrayObject *py_chg_leak_interp = 
              (PyArrayObject *) PyArray_SimpleNew(2, chg_leak_dim, NPY_DOUBLE);
  PyArrayObject *py_chg_open_interp = 
              (PyArrayObject *) PyArray_SimpleNew(2, chg_leak_dim, NPY_DOUBLE);
  if (!py_chg_leak_interp || !py_chg_open_interp) {
    return NULL;
  }
  
  /* put arguments into variables */
  if (!PyArg_ParseTuple(args, "OO", &opy_chg_leak, &opy_psi_node)) {
    return NULL;
  }
  
  py_psi_node = (PyArrayObject *) PyArray_FROMANY(opy_psi_node, NPY_INT, 1, 1, NPY_IN_ARRAY);
  py_chg_leak = (PyArrayObject *) PyArray_FROMANY(opy_chg_leak, NPY_DOUBLE, 2, 2, NPY_IN_ARRAY);
  if (!py_psi_node || !py_chg_leak) {
    return NULL;
  }
  
  /* call InterpolatePsi */
  status = InterpolatePsi((double *) PyArray_DATA(py_chg_leak), 
                          (int *) PyArray_DATA(py_psi_node), 
                          (double *) PyArray_DATA(py_chg_leak_interp),
                          (double *) PyArray_DATA(py_chg_open_interp));
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in InterpolatePsi.");
    return NULL;
  }
  
  Py_DECREF(py_psi_node);
  Py_DECREF(py_chg_leak);
  
  return Py_BuildValue("NN",py_chg_leak_interp, py_chg_open_interp);
}


static PyObject * py_InterpolatePhi(PyObject *self, PyObject *args) {
  /* input variables */
  PyObject *opy_dtde_l, *opy_q_dtde;
  PyArrayObject *py_dtde_l, *py_q_dtde;
  int shft_nit;
  
  /* local variables */
  int status;
  
  /* return variables */
  npy_intp phi_max_dim[] = {MAX_PHI};
  PyArrayObject *py_dtde_q = 
    (PyArrayObject *) PyArray_SimpleNew(1, phi_max_dim, NPY_DOUBLE);
  if (!py_dtde_q) {
    return NULL;
  }
 
  /* put arguments into variables */
  if (!PyArg_ParseTuple(args, "OOi", &opy_dtde_l, &opy_q_dtde, &shft_nit)) {
    return NULL;
  }
   
  py_dtde_l = (PyArrayObject *) PyArray_FROMANY(opy_dtde_l, NPY_DOUBLE, 1, 1, NPY_IN_ARRAY);
  py_q_dtde = (PyArrayObject *) PyArray_FROMANY(opy_q_dtde, NPY_INT, 1, 1, NPY_IN_ARRAY);
  if (!py_dtde_l || !py_q_dtde) {
    return NULL;
  }
  
  /* call InterpolatePhi */
  status = InterpolatePhi((double *) PyArray_DATA(py_dtde_l),
                          (int *) PyArray_DATA(py_q_dtde), shft_nit,
                          (double *) PyArray_DATA(py_dtde_q));
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in InterpolatePhi.");
    return NULL;
  }
  
  Py_DECREF(py_dtde_l);
  Py_DECREF(py_q_dtde);
  
  return Py_BuildValue("N",py_dtde_q);
}


static PyObject * py_FillLevelArrays(PyObject *self, PyObject *args) {
  /* input variables */
  PyObject *opy_chg_leak_kt, *opy_chg_open_kt, *opy_dtde_q, *opy_levels;
  PyArrayObject *py_chg_leak_kt, *py_chg_open_kt, *py_dtde_q, *py_levels;
  
  /* local variables */
  int status;
  
  /* return variables */
  npy_intp lt_dim[] = {MAX_TAIL_LEN, NUM_LEV};
  npy_intp l_dim[] = {NUM_LEV};
  PyArrayObject *py_chg_leak_lt = 
    (PyArrayObject *) PyArray_SimpleNew(2, lt_dim, NPY_DOUBLE);
  PyArrayObject *py_chg_open_lt = 
    (PyArrayObject *) PyArray_SimpleNew(2, lt_dim, NPY_DOUBLE);
  PyArrayObject *py_dpde_l = 
    (PyArrayObject *) PyArray_SimpleNew(1, l_dim, NPY_DOUBLE);
  if (!py_chg_leak_lt || !py_chg_open_lt || !py_dpde_l) {
    return NULL;
  }
  
  /* put input arguments into variables */
  if (!PyArg_ParseTuple(args, "OOOO", &opy_chg_leak_kt, &opy_chg_open_kt,
                                      &opy_dtde_q, &opy_levels)) {
    return NULL;
  }
  
  py_chg_leak_kt = 
    (PyArrayObject *) PyArray_FROMANY(opy_chg_leak_kt, NPY_DOUBLE, 2, 2, NPY_IN_ARRAY);
  py_chg_open_kt = 
    (PyArrayObject *) PyArray_FROMANY(opy_chg_open_kt, NPY_DOUBLE, 2, 2, NPY_IN_ARRAY);
  py_dtde_q = (PyArrayObject *) PyArray_FROMANY(opy_dtde_q, NPY_DOUBLE, 1, 1, NPY_IN_ARRAY);
  py_levels = (PyArrayObject *) PyArray_FROMANY(opy_levels, NPY_INT, 1, 1, NPY_IN_ARRAY);
  if (!py_chg_leak_kt || !py_chg_open_kt || !py_dtde_q || !py_levels) {
    return NULL;
  }
  
  /* call FillLevelArrays */
  status = FillLevelArrays((double *) PyArray_DATA(py_chg_leak_kt),
                           (double *) PyArray_DATA(py_chg_open_kt),
                           (double *) PyArray_DATA(py_dtde_q),
                           (int *) PyArray_DATA(py_levels),
                           (double *) PyArray_DATA(py_chg_leak_lt),
                           (double *) PyArray_DATA(py_chg_open_lt),
                           (double *) PyArray_DATA(py_dpde_l));
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in FillLevelArrays.");
    return NULL;
  }
  
  Py_DECREF(py_chg_leak_kt);
  Py_DECREF(py_chg_open_kt);
  Py_DECREF(py_dtde_q);
  Py_DECREF(py_levels);
  
  return Py_BuildValue("NNN",py_chg_leak_lt, py_chg_open_lt, py_dpde_l);
}


static PyObject * py_DecomposeRN(PyObject *self, PyObject *args) {
  /* input variables */
  PyObject *opy_data;
  PyArrayObject *py_data;
  double pclip;
  int noise_model;
  
  /* local variables */
  int status;
  int arrx, arry;
  
  /* return variables */
  npy_intp * out_dim;
  PyArrayObject * py_sig;
  PyArrayObject * py_noise;
  
  /* put arguments into variables */
  if (!PyArg_ParseTuple(args, "Odi", &opy_data, &pclip, &noise_model)) {
    return NULL;
  }
  
  py_data = (PyArrayObject *) PyArray_FROMANY(opy_data, NPY_DOUBLE, 2, 2, NPY_IN_ARRAY);
  if (!py_data) {
    return NULL;
  }
  
  /* assign/allocate local variables */
  arrx = py_data->dimensions[0];
  arry = py_data->dimensions[1];
  
  /* return variables */
  out_dim = (npy_intp *) malloc(2 * sizeof(npy_intp));
  out_dim[0] = (npy_intp) arrx;
  out_dim[1] = (npy_intp) arry;  
  py_sig = (PyArrayObject *) PyArray_SimpleNew(2, out_dim, NPY_DOUBLE);
  py_noise = (PyArrayObject *) PyArray_SimpleNew(2, out_dim, NPY_DOUBLE);
  if (!py_sig || !py_noise) {
    return NULL;
  }
  
  /* call DecomposeRN */
  status = DecomposeRN(arrx, arry, (double *) PyArray_DATA(py_data), pclip,
                       noise_model, (double *) PyArray_DATA(py_sig), 
                       (double *) PyArray_DATA(py_noise));
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in DecomposeRN.");
    return NULL;
  }
  
  free(out_dim);
  
  Py_DECREF(py_data);
  
  return Py_BuildValue("NN", py_sig, py_noise);
}


static PyObject * py_FixYCte(PyObject *self, PyObject *args) {
  /* input variables */
  PyObject *opy_sig_cte, *opy_levels, *opy_dpde_l;
  PyObject *opy_cte_frac, *opy_chg_leak_lt, *opy_chg_open_lt;
  PyArrayObject *py_sig_cte, *py_levels, *py_dpde_l;
  PyArrayObject *py_cte_frac, *py_chg_leak_lt, *py_chg_open_lt;
  int sim_nit, shft_nit;
  double sub_thresh;
  
  /* local variables */
  int status;
  int arrx, arry;
  
  /* return variables */
  npy_intp * out_dim;
  PyArrayObject * py_sig_cor;
  
  /* put arguments into variables */
  if (!PyArg_ParseTuple(args, "OiidOOOOO", &opy_sig_cte, &sim_nit, &shft_nit,
                        &sub_thresh, &opy_cte_frac, &opy_levels, &opy_dpde_l, 
                        &opy_chg_leak_lt, &opy_chg_open_lt)) {
    return NULL;
  }
  
  py_sig_cte = (PyArrayObject *) PyArray_FROMANY(opy_sig_cte, NPY_DOUBLE,
                                                 2, 2, NPY_IN_ARRAY);
  py_cte_frac = (PyArrayObject *) PyArray_FROMANY(opy_cte_frac, NPY_DOUBLE,
                                                  2, 2, NPY_IN_ARRAY);
  py_levels = (PyArrayObject *) PyArray_FROMANY(opy_levels, NPY_INT,
                                                1, 1, NPY_IN_ARRAY);
  py_dpde_l = (PyArrayObject *) PyArray_FROMANY(opy_dpde_l, NPY_DOUBLE,
                                                1, 1, NPY_IN_ARRAY);
  py_chg_leak_lt = (PyArrayObject *) PyArray_FROMANY(opy_chg_leak_lt, NPY_DOUBLE, 
                                                     2, 2, NPY_IN_ARRAY);
  py_chg_open_lt = (PyArrayObject *) PyArray_FROMANY(opy_chg_open_lt, NPY_DOUBLE, 
                                                     2, 2, NPY_IN_ARRAY);
  if (!py_sig_cte || !py_cte_frac || !py_levels || !py_dpde_l || 
      !py_chg_leak_lt || !py_chg_open_lt) {
    return NULL;
  }
  
  /* local variables */
  arrx = py_sig_cte->dimensions[0];
  arry = py_sig_cte->dimensions[1];
  
  /* return variables */
  out_dim = (npy_intp *) malloc(2 * sizeof(npy_intp));
  out_dim[0] = (npy_intp) arrx;
  out_dim[1] = (npy_intp) arry;
  py_sig_cor = (PyArrayObject *) PyArray_SimpleNew(2, out_dim, NPY_DOUBLE);
  if (!py_sig_cor) {
    return NULL;
  }
  
  /* call FixYCte */
  status = FixYCte(arrx, arry, (double *) PyArray_DATA(py_sig_cte),
                   (double *) PyArray_DATA(py_sig_cor), sim_nit,
                   shft_nit, sub_thresh, (double *) PyArray_DATA(py_cte_frac),
                   (int *) PyArray_DATA(py_levels),
                   (double *) PyArray_DATA(py_dpde_l),
                   (double *) PyArray_DATA(py_chg_leak_lt),
                   (double *) PyArray_DATA(py_chg_open_lt));
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in FixYCte.");
    return NULL;
  }
  
  free(out_dim);
  
  Py_DECREF(py_sig_cte);
  Py_DECREF(py_cte_frac);
  Py_DECREF(py_levels);
  Py_DECREF(py_dpde_l);
  Py_DECREF(py_chg_leak_lt);
  Py_DECREF(py_chg_open_lt);
  
  return Py_BuildValue("N", py_sig_cor);
}


static PyObject * py_AddYCte(PyObject *self, PyObject *args) {
  /* input variables */
  PyObject *opy_sig_cte, *opy_levels, *opy_dpde_l;
  PyObject *opy_cte_frac, *opy_chg_leak_lt, *opy_chg_open_lt;
  PyArrayObject *py_sig_cte, *py_levels, *py_dpde_l;
  PyArrayObject *py_cte_frac, *py_chg_leak_lt, *py_chg_open_lt;
  int shft_nit;
  
  /* local variables */
  int status;
  int arrx, arry;
  
  /* return variables */
  npy_intp * out_dim;
  PyArrayObject * py_sig_cor;
  
  /* put arguments into variables */
  if (!PyArg_ParseTuple(args, "OiOOOOO", &opy_sig_cte, &shft_nit, 
                        &opy_cte_frac, &opy_levels, &opy_dpde_l, 
                        &opy_chg_leak_lt, &opy_chg_open_lt)) {
    return NULL;
  }
  
  py_sig_cte = (PyArrayObject *) PyArray_FROMANY(opy_sig_cte, NPY_DOUBLE,
                                                 2, 2, NPY_IN_ARRAY);
  py_cte_frac = (PyArrayObject *) PyArray_FROMANY(opy_cte_frac, NPY_DOUBLE,
                                                  2, 2, NPY_IN_ARRAY);
  py_levels = (PyArrayObject *) PyArray_FROMANY(opy_levels, NPY_INT,
                                                1, 1, NPY_IN_ARRAY);
  py_dpde_l = (PyArrayObject *) PyArray_FROMANY(opy_dpde_l, NPY_DOUBLE,
                                                1, 1, NPY_IN_ARRAY);
  py_chg_leak_lt = (PyArrayObject *) PyArray_FROMANY(opy_chg_leak_lt, NPY_DOUBLE, 
                                                     2, 2, NPY_IN_ARRAY);
  py_chg_open_lt = (PyArrayObject *) PyArray_FROMANY(opy_chg_open_lt, NPY_DOUBLE, 
                                                     2, 2, NPY_IN_ARRAY);
  if (!py_sig_cte || !py_cte_frac || !py_levels || !py_dpde_l || 
      !py_chg_leak_lt || !py_chg_open_lt) {
    return NULL;
  }
  
  /* local variables */
  arrx = py_sig_cte->dimensions[0];
  arry = py_sig_cte->dimensions[1];
  
  /* return variables */
  out_dim = (npy_intp *) malloc(2 * sizeof(npy_intp));
  out_dim[0] = (npy_intp) arrx;
  out_dim[1] = (npy_intp) arry;
  py_sig_cor = (PyArrayObject *) PyArray_SimpleNew(2, out_dim, NPY_DOUBLE);
  if (!py_sig_cor) {
    return NULL;
  }
  
  /* call FixYCte */
  status = AddYCte(arrx, arry, (double *) PyArray_DATA(py_sig_cte),
                   (double *) PyArray_DATA(py_sig_cor), shft_nit,
                   (double *) PyArray_DATA(py_cte_frac),
                   (int *) PyArray_DATA(py_levels),
                   (double *) PyArray_DATA(py_dpde_l),
                   (double *) PyArray_DATA(py_chg_leak_lt),
                   (double *) PyArray_DATA(py_chg_open_lt));
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in AddYCte.");
    return NULL;
  }
  
  free(out_dim);
  
  Py_DECREF(py_sig_cte);
  Py_DECREF(py_cte_frac);
  Py_DECREF(py_levels);
  Py_DECREF(py_dpde_l);
  Py_DECREF(py_chg_leak_lt);
  Py_DECREF(py_chg_open_lt);
  
  return Py_BuildValue("N", py_sig_cor);
}

static PyMethodDef PixCte_FixY_methods[] =
{
  {"CalcCteFrac",  py_CalcCteFrac, METH_VARARGS, "Calculate CTE multiplier."},
  {"InterpolatePsi", py_InterpolatePsi, METH_VARARGS, "Interpolate to full tail profile."},
  {"InterpolatePhi", py_InterpolatePhi, METH_VARARGS, "Interpolate charge traps."},
  {"FillLevelArrays", py_FillLevelArrays, METH_VARARGS, "FillLevelArrays."},
  {"DecomposeRN", py_DecomposeRN, METH_VARARGS, "Separate signal and readnoise."},
  {"FixYCte", py_FixYCte, METH_VARARGS, "Perform parallel CTE correction."},
  {"AddYCte", py_AddYCte, METH_VARARGS, "Add parallel CTE distortion."},
  {NULL, NULL, 0, NULL} /* sentinel */
  
};

PyMODINIT_FUNC initPixCte_FixY(void)
{
  (void) Py_InitModule("PixCte_FixY", PixCte_FixY_methods);
  import_array(); // Must be present for NumPy
}
