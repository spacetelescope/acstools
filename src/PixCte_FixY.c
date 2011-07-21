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
                                                    1, 1, NPY_ARRAY_IN_ARRAY);
  py_scale_val = (PyArrayObject *) PyArray_FROMANY(opy_scale_val, NPY_DOUBLE, 
                                                    1, 1, NPY_ARRAY_IN_ARRAY);
  
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
  int status, i, j;
  int psi_node[NUM_PSI];
  double chg_leak[NUM_PSI*NUM_LOGQ];
  double chg_leak_interp[MAX_TAIL_LEN*NUM_LOGQ];
  double chg_open_interp[MAX_TAIL_LEN*NUM_LOGQ];
  
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
  
  py_psi_node = (PyArrayObject *) PyArray_FROMANY(opy_psi_node, NPY_INT, 1, 1, NPY_ARRAY_IN_ARRAY);
  py_chg_leak = (PyArrayObject *) PyArray_FROMANY(opy_chg_leak, NPY_DOUBLE, 2, 2, NPY_ARRAY_IN_ARRAY);
  if (!py_psi_node || !py_chg_leak) {
    return NULL;
  }
  
  /* put array object inputs into local C arrays */
  for (i = 0; i < NUM_PSI; i++) {
    psi_node[i] = *(int *) PyArray_GETPTR1(py_psi_node, i);
    
    for (j = 0; j < NUM_LOGQ; j++) {
      chg_leak[i*NUM_LOGQ + j] = *(double *) PyArray_GETPTR2(py_chg_leak, i, j);
    }
  }
  
  /* call InterpolatePsi */
  status = InterpolatePsi(chg_leak, psi_node, chg_leak_interp, chg_open_interp);
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in InterpolatePsi.");
    return NULL;
  }
  
  /* copy chg_leak_interp and chg_open_interp to returned PyArrayObjects */
  for (i = 0; i < MAX_TAIL_LEN; i++) {
    for (j = 0; j < NUM_LOGQ; j++) {
      *(npy_double *) PyArray_GETPTR2(py_chg_leak_interp, i, j) = 
                                                chg_leak_interp[i*NUM_LOGQ + j];
      *(npy_double *) PyArray_GETPTR2(py_chg_open_interp, i, j) = 
                                                chg_open_interp[i*NUM_LOGQ + j];
    }
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
  int status, p;
  double dtde_l[NUM_PHI];
  int q_dtde[NUM_PHI];
  double dtde_q[MAX_PHI];
  
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
   
  py_dtde_l = (PyArrayObject *) PyArray_FROMANY(opy_dtde_l, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
  py_q_dtde = (PyArrayObject *) PyArray_FROMANY(opy_q_dtde, NPY_INT, 1, 1, NPY_ARRAY_IN_ARRAY);
  if (!py_dtde_l || !py_q_dtde) {
    return NULL;
  }
  
  /* copy input python array to local C array */
  for (p = 0; p < NUM_PHI; p++) {
    dtde_l[p] = *(double *) PyArray_GETPTR1(py_dtde_l, p);
    q_dtde[p] = *(int *) PyArray_GETPTR1(py_q_dtde, p);
  }
  
  /* call InterpolatePhi */
  status = InterpolatePhi(dtde_l, q_dtde, shft_nit, dtde_q);
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in InterpolatePhi.");
    return NULL;
  }
  
  /* copy local C arrays to return variables */
  for (p = 0; p < MAX_PHI; p++) {
    *(npy_double *) PyArray_GETPTR1(py_dtde_q, p) = dtde_q[p];
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
  int status, i, j;
  double chg_leak_kt[MAX_TAIL_LEN*NUM_LOGQ];
  double chg_open_kt[MAX_TAIL_LEN*NUM_LOGQ];
  double dtde_q[MAX_PHI];
  int levels[NUM_LEV];
  double chg_leak_lt[MAX_TAIL_LEN*NUM_LEV];
  double chg_open_lt[MAX_TAIL_LEN*NUM_LEV];
  double dpde_l[NUM_LEV];
  int tail_len[NUM_LEV];
  
  /* return variables */
  npy_intp lt_dim[] = {MAX_TAIL_LEN, NUM_LEV};
  npy_intp l_dim[] = {NUM_LEV};
  PyArrayObject *py_chg_leak_lt = 
    (PyArrayObject *) PyArray_SimpleNew(2, lt_dim, NPY_DOUBLE);
  PyArrayObject *py_chg_open_lt = 
    (PyArrayObject *) PyArray_SimpleNew(2, lt_dim, NPY_DOUBLE);
  PyArrayObject *py_dpde_l = 
    (PyArrayObject *) PyArray_SimpleNew(1, l_dim, NPY_DOUBLE);
  PyArrayObject *py_tail_len = 
    (PyArrayObject *) PyArray_SimpleNew(1, l_dim, NPY_INT);
  if (!py_chg_leak_lt || !py_chg_open_lt || !py_dpde_l || !py_tail_len) {
    return NULL;
  }
  
  /* put input arguments into variables */
  if (!PyArg_ParseTuple(args, "OOOO", &opy_chg_leak_kt, &opy_chg_open_kt,
                                      &opy_dtde_q, &opy_levels)) {
    return NULL;
  }
  
  py_chg_leak_kt = 
    (PyArrayObject *) PyArray_FROMANY(opy_chg_leak_kt, NPY_DOUBLE, 2, 2, NPY_ARRAY_IN_ARRAY);
  py_chg_open_kt = 
    (PyArrayObject *) PyArray_FROMANY(opy_chg_open_kt, NPY_DOUBLE, 2, 2, NPY_ARRAY_IN_ARRAY);
  py_dtde_q = (PyArrayObject *) PyArray_FROMANY(opy_dtde_q, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
  py_levels = (PyArrayObject *) PyArray_FROMANY(opy_levels, NPY_INT, 1, 1, NPY_ARRAY_IN_ARRAY);
  if (!py_chg_leak_kt || !py_chg_open_kt || !py_dtde_q || !py_levels) {
    return NULL;
  }
  
  /* copy input python arrays to local C arrays */
  for (i = 0; i < MAX_PHI; i++) {
    dtde_q[i] = (double) *(npy_double *) PyArray_GETPTR1(py_dtde_q, i);
  }
  
  for (i = 0; i < NUM_LEV; i++) {
    levels[i] = (int) *(npy_int *) PyArray_GETPTR1(py_levels, i);
  }
  
  for (i = 0; i < MAX_TAIL_LEN; i++) {
    for (j = 0; j < NUM_LOGQ; j++) {
      chg_leak_kt[i*NUM_LOGQ + j] = (double) *(npy_double *) PyArray_GETPTR2(py_chg_leak_kt, i, j);
      chg_open_kt[i*NUM_LOGQ + j] = (double) *(npy_double *) PyArray_GETPTR2(py_chg_open_kt, i, j);
    }
  }
  
  /* call FillLevelArrays */
  status = FillLevelArrays(chg_leak_kt, chg_open_kt, dtde_q, levels,
                           chg_leak_lt, chg_open_lt, dpde_l, tail_len);
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in FillLevelArrays.");
    return NULL;
  }
  
  /* copy local C arrays to return variables */
  for (j = 0; j < NUM_LEV; j++) {
    *(npy_double *) PyArray_GETPTR1(py_dpde_l, j) = (npy_double) dpde_l[j];
    *(npy_int *) PyArray_GETPTR1(py_tail_len, j) = (npy_int) tail_len[j];
    
    for (i = 0; i < MAX_TAIL_LEN; i++) {
      *(npy_double *) PyArray_GETPTR2(py_chg_leak_lt, i, j) = 
        (npy_double) chg_leak_lt[i*NUM_LEV + j];
      *(npy_double *) PyArray_GETPTR2(py_chg_open_lt, i, j) = 
        (npy_double) chg_open_lt[i*NUM_LEV + j];
    }
  }
  
  Py_DECREF(py_chg_leak_kt);
  Py_DECREF(py_chg_open_kt);
  Py_DECREF(py_dtde_q);
  Py_DECREF(py_levels);
  
  return Py_BuildValue("NNNN",py_chg_leak_lt, py_chg_open_lt, py_dpde_l, py_tail_len);
}

static PyObject * py_DecomposeRN(PyObject *self, PyObject *args) {
  /* input variables */
  PyObject *opy_data;
  PyArrayObject *py_data;
  double pclip;
  
  /* local variables */
  int status, i, j;
  int arrx, arry;
  double * data;
  double * sig;
  double * noise;
  
  /* return variables */
  npy_intp * out_dim;
  PyArrayObject * py_sig;
  PyArrayObject * py_noise;
  
  /* put arguments into variables */
  if (!PyArg_ParseTuple(args, "Od", &opy_data, &pclip)) {
    return NULL;
  }
  
  py_data = (PyArrayObject *) PyArray_FROMANY(opy_data, NPY_DOUBLE, 2, 2, NPY_ARRAY_IN_ARRAY);
  if (!py_data) {
    return NULL;
  }
  
  /* assign/allocate local variables */
  arrx = py_data->dimensions[0];
  arry = py_data->dimensions[1];
  data = (double *) malloc(arrx * arry * sizeof(double));
  sig = (double *) malloc(arrx * arry * sizeof(double));
  noise = (double *) malloc(arrx * arry * sizeof(double));
  
  /* return variables */
  out_dim = (npy_intp *) malloc(2 * sizeof(npy_intp));
  out_dim[0] = (npy_intp) arrx;
  out_dim[1] = (npy_intp) arry;  
  py_sig = (PyArrayObject *) PyArray_SimpleNew(2, out_dim, NPY_DOUBLE);
  py_noise = (PyArrayObject *) PyArray_SimpleNew(2, out_dim, NPY_DOUBLE);
  if (!py_sig || !py_noise) {
    return NULL;
  }
  
  /* copy input Python arrays to local C arrays */
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      data[i*arry + j] = (double) *(npy_double *) PyArray_GETPTR2(py_data, i, j);
    }
  }
  
  /* call DecomposeRN */
  status = DecomposeRN(arrx, arry, data, pclip, sig, noise);
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in DecomposeRN.");
    return NULL;
  }
  
  /* copy local C arrays to return Python arrays */
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      *(npy_double *) PyArray_GETPTR2(py_sig, i, j) = (npy_double) sig[i*arry + j];
      *(npy_double *) PyArray_GETPTR2(py_noise, i, j) = (npy_double) noise[i*arry + j];
    }
  }
  
  free(out_dim);
  free(data);
  free(sig);
  free(noise);
  
  Py_DECREF(py_data);
  
  return Py_BuildValue("NN", py_sig, py_noise);
}

static PyObject * py_FixYCte(PyObject *self, PyObject *args) {
  /* input variables */
  PyObject *opy_sig_cte, *opy_levels, *opy_dpde_l, *opy_tail_len;
  PyObject *opy_chg_leak_lt, *opy_chg_open_lt;
  PyArrayObject *py_sig_cte, *py_levels, *py_dpde_l, *py_tail_len;
  PyArrayObject *py_chg_leak_lt, *py_chg_open_lt;
  double cte_frac;
  int sim_nit, shft_nit;
  const char *amp_name, *log_file;
  
  /* local variables */
  int status, i, j;
  int arrx, arry;
  double * sig_cte;
  double * sig_cor;
  int levels[NUM_LEV];
  double dpde_l[NUM_LEV];
  int tail_len[NUM_LEV];
  double chg_leak_lt[MAX_TAIL_LEN*NUM_LEV];
  double chg_open_lt[MAX_TAIL_LEN*NUM_LEV];
  
  /* return variables */
  npy_intp * out_dim;
  PyArrayObject * py_sig_cor;
  
  /* put arguments into variables */
  if (!PyArg_ParseTuple(args, "OdiiOOOOOss", &opy_sig_cte, &cte_frac, &sim_nit, 
                        &shft_nit, &opy_levels, &opy_dpde_l, & opy_tail_len, 
                        &opy_chg_leak_lt, &opy_chg_open_lt, &amp_name, &log_file)) {
    return NULL;
  }
  
  py_sig_cte = (PyArrayObject *) PyArray_FROMANY(opy_sig_cte, NPY_DOUBLE, 2, 2, NPY_ARRAY_IN_ARRAY);
  py_levels = (PyArrayObject *) PyArray_FROMANY(opy_levels, NPY_INT, 1, 1, NPY_ARRAY_IN_ARRAY);
  py_dpde_l = (PyArrayObject *) PyArray_FROMANY(opy_dpde_l, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
  py_tail_len = (PyArrayObject *) PyArray_FROMANY(opy_tail_len, NPY_INT, 1, 1, NPY_ARRAY_IN_ARRAY);
  py_chg_leak_lt = (PyArrayObject *) PyArray_FROMANY(opy_chg_leak_lt, NPY_DOUBLE, 
                                                     2, 2, NPY_ARRAY_IN_ARRAY);
  py_chg_open_lt = (PyArrayObject *) PyArray_FROMANY(opy_chg_open_lt, NPY_DOUBLE, 
                                                     2, 2, NPY_ARRAY_IN_ARRAY);
  if (!py_sig_cte || !py_levels || !py_dpde_l || !py_tail_len || 
      !py_chg_leak_lt || !py_chg_open_lt) {
    return NULL;
  }
  
  /* local variables */
  arrx = py_sig_cte->dimensions[0];
  arry = py_sig_cte->dimensions[1];
  sig_cte = (double *) malloc(arrx * arry * sizeof(double));
  sig_cor = (double *) malloc(arrx * arry * sizeof(double));
  
  /* return variables */
  out_dim = (npy_intp *) malloc(2 * sizeof(npy_intp));
  out_dim[0] = (npy_intp) arrx;
  out_dim[1] = (npy_intp) arry;
  py_sig_cor = (PyArrayObject *) PyArray_SimpleNew(2, out_dim, NPY_DOUBLE);
  if (!py_sig_cor) {
    return NULL;
  }
  
  /* copy input Python arrays to local C arrays */  
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      sig_cte[i*arry + j] = (double) *(npy_double *) PyArray_GETPTR2(py_sig_cte, i, j);
    }
  }
  
  for (i = 0; i < NUM_LEV; i++) {
    levels[i] = (int) *(npy_int *) PyArray_GETPTR1(py_levels, i);
    tail_len[i] = (int) *(npy_int *) PyArray_GETPTR1(py_tail_len, i);
    dpde_l[i] = (double) *(npy_double *) PyArray_GETPTR1(py_dpde_l, i);
    
    for (j = 0; j < MAX_TAIL_LEN; j++) {
      chg_leak_lt[j*NUM_LEV + i] = 
        (double) *(npy_double *) PyArray_GETPTR2(py_chg_leak_lt, j, i);
      chg_open_lt[j*NUM_LEV + i] = 
        (double) *(npy_double *) PyArray_GETPTR2(py_chg_open_lt, j, i);
    }
  }
  
  /* call FixYCte */
  status = FixYCte(arrx, arry, sig_cte, sig_cor, cte_frac, sim_nit, shft_nit,
                   levels, dpde_l, tail_len, chg_leak_lt, chg_open_lt, amp_name, log_file);
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in FixYCte.");
    return NULL;
  }
  
  /* copy fixed data in C array to output Python array */
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      *(double *) PyArray_GETPTR2(py_sig_cor, i, j) = sig_cor[i*arry + j];
    }
  }
  
  free(out_dim);
  free(sig_cte);
  free(sig_cor);
  
  Py_DECREF(py_sig_cte);
  Py_DECREF(py_levels);
  Py_DECREF(py_tail_len);
  Py_DECREF(py_dpde_l);
  Py_DECREF(py_chg_leak_lt);
  Py_DECREF(py_chg_open_lt);
  
  return Py_BuildValue("N", py_sig_cor);
}

static PyObject * py_AddYCte(PyObject *self, PyObject *args) {
  /* input variables */
  PyObject *opy_sig_cte, *opy_levels, *opy_dpde_l, *opy_tail_len;
  PyObject *opy_chg_leak_lt, *opy_chg_open_lt;
  PyArrayObject *py_sig_cte, *py_levels, *py_dpde_l, *py_tail_len;
  PyArrayObject *py_chg_leak_lt, *py_chg_open_lt;
  double cte_frac;
  int shft_nit;
  
  /* local variables */
  int status, i, j;
  int arrx, arry;
  double * sig_cte;
  double * sig_cor;
  int levels[NUM_LEV];
  double dpde_l[NUM_LEV];
  int tail_len[NUM_LEV];
  double chg_leak_lt[MAX_TAIL_LEN*NUM_LEV];
  double chg_open_lt[MAX_TAIL_LEN*NUM_LEV];
  
  /* return variables */
  npy_intp * out_dim;
  PyArrayObject * py_sig_cor;
  
  /* put arguments into variables */
  if (!PyArg_ParseTuple(args, "OdiOOOOO", &opy_sig_cte, &cte_frac, 
                        &shft_nit, &opy_levels, &opy_dpde_l, & opy_tail_len, 
                        &opy_chg_leak_lt, &opy_chg_open_lt)) {
    return NULL;
  }
  
  py_sig_cte = (PyArrayObject *) PyArray_FROMANY(opy_sig_cte, NPY_DOUBLE, 2, 2, NPY_ARRAY_IN_ARRAY);
  py_levels = (PyArrayObject *) PyArray_FROMANY(opy_levels, NPY_INT, 1, 1, NPY_ARRAY_IN_ARRAY);
  py_dpde_l = (PyArrayObject *) PyArray_FROMANY(opy_dpde_l, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
  py_tail_len = (PyArrayObject *) PyArray_FROMANY(opy_tail_len, NPY_INT, 1, 1, NPY_ARRAY_IN_ARRAY);
  py_chg_leak_lt = (PyArrayObject *) PyArray_FROMANY(opy_chg_leak_lt, NPY_DOUBLE, 
                                                     2, 2, NPY_ARRAY_IN_ARRAY);
  py_chg_open_lt = (PyArrayObject *) PyArray_FROMANY(opy_chg_open_lt, NPY_DOUBLE, 
                                                     2, 2, NPY_ARRAY_IN_ARRAY);
  if (!py_sig_cte || !py_levels || !py_dpde_l || !py_tail_len || 
      !py_chg_leak_lt || !py_chg_open_lt) {
    return NULL;
  }
  
  /* local variables */
  arrx = py_sig_cte->dimensions[0];
  arry = py_sig_cte->dimensions[1];
  sig_cte = (double *) malloc(arrx * arry * sizeof(double));
  sig_cor = (double *) malloc(arrx * arry * sizeof(double));
  
  /* return variables */
  out_dim = (npy_intp *) malloc(2 * sizeof(npy_intp));
  out_dim[0] = (npy_intp) arrx;
  out_dim[1] = (npy_intp) arry;
  py_sig_cor = (PyArrayObject *) PyArray_SimpleNew(2, out_dim, NPY_DOUBLE);
  if (!py_sig_cor) {
    return NULL;
  }
  
  /* copy input Python arrays to local C arrays */  
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      sig_cte[i*arry + j] = (double) *(npy_double *) PyArray_GETPTR2(py_sig_cte, i, j);
    }
  }
  
  for (i = 0; i < NUM_LEV; i++) {
    levels[i] = (int) *(npy_int *) PyArray_GETPTR1(py_levels, i);
    tail_len[i] = (int) *(npy_int *) PyArray_GETPTR1(py_tail_len, i);
    dpde_l[i] = (double) *(npy_double *) PyArray_GETPTR1(py_dpde_l, i);
    
    for (j = 0; j < MAX_TAIL_LEN; j++) {
      chg_leak_lt[j*NUM_LEV + i] = 
      (double) *(npy_double *) PyArray_GETPTR2(py_chg_leak_lt, j, i);
      chg_open_lt[j*NUM_LEV + i] = 
      (double) *(npy_double *) PyArray_GETPTR2(py_chg_open_lt, j, i);
    }
  }
  
  /* call FixYCte */
  status = AddYCte(arrx, arry, sig_cte, sig_cor, cte_frac, shft_nit,
                   levels, dpde_l, tail_len, chg_leak_lt, chg_open_lt);
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in AddYCte.");
    return NULL;
  }
  
  /* copy fixed data in C array to output Python array */
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      *(double *) PyArray_GETPTR2(py_sig_cor, i, j) = sig_cor[i*arry + j];
    }
  }
  
  free(out_dim);
  free(sig_cte);
  free(sig_cor);
  
  Py_DECREF(py_sig_cte);
  Py_DECREF(py_levels);
  Py_DECREF(py_tail_len);
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
