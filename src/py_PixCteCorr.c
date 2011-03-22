#import <stdlib.h>
#import <stdio.h>

#include <Python.h>
#include <numpy/arrayobject.h>

#include "PixCteCorr.h"

static PyObject * py_CalcCteFrac(PyObject *self, PyObject *args) {
  /* input variables */
  const double mjd;
  const int instrument;
  
  /* local variables */
  double cte_frac;
  
  /* put arguments into variables */
  if (!PyArg_ParseTuple(args, "di", &mjd, &instrument)) {
    return NULL;
  }
  
  /* get cte_frac */
  cte_frac = CalcCteFrac(mjd, instrument);
  
  /* test whether it's good */
  if (cte_frac < 0) {
    PyErr_SetString(PyExc_ValueError,"Instrument is invalid for CalcCteFrac.");
    return NULL;
  }
  
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
  
  /* return variable */
  npy_intp chg_leak_dim[] = {MAX_TAIL_LEN, NUM_LOGQ};
  PyArrayObject *py_chg_leak_interp = (PyArrayObject *) PyArray_SimpleNew(2, chg_leak_dim, NPY_DOUBLE);
  if (!py_chg_leak_interp) {
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
  
  /* put array object inputs into local C arrays */
  for (i = 0; i < NUM_PSI; i++) {
    psi_node[i] = *(int *) PyArray_GETPTR1(py_psi_node, i);
    
    for (j = 0; j < NUM_LOGQ; j++) {
      chg_leak[i*NUM_LOGQ + j] = *(double *) PyArray_GETPTR2(py_chg_leak, i, j);
    }
  }
  
  /* call InterpolatePsi */
  status = InterpolatePsi(chg_leak, psi_node, chg_leak_interp);
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in InterpolatePsi.");
    return NULL;
  }
  
  /* copy chg_leak_interp to returned PyArrayObject */
  for (i = 0; i < MAX_TAIL_LEN; i++) {
    for (j = 0; j < NUM_LOGQ; j++) {
      *(npy_double *) PyArray_GETPTR2(py_chg_leak_interp, i, j) = chg_leak_interp[i*NUM_LOGQ + j];
    }
  }
  
  Py_DECREF(py_psi_node);
  Py_DECREF(py_chg_leak);
  
  return Py_BuildValue("O",py_chg_leak_interp);
}

static PyObject * py_InterpolatePhi(PyObject *self, PyObject *args) {
  /* input variables */
  PyObject *opy_dtde_l;
  PyArrayObject *py_dtde_l;
  double cte_frac;
  
  /* local variables */
  int status, p;
  double dtde_l[NUM_PHI];
  double dtde_q[MAX_PHI];
  int q_pix_array[MAX_PHI];
  int pix_q_array[MAX_PHI] = {0};
  int ycte_qmax;
  
  /* return variables */
  npy_intp phi_dim[] = {NUM_PHI};
  npy_intp phi_max_dim[] = {MAX_PHI};
  PyArrayObject *py_dtde_q = 
    (PyArrayObject *) PyArray_SimpleNew(1, phi_max_dim, NPY_DOUBLE);
  PyArrayObject *py_q_pix_array = 
    (PyArrayObject *) PyArray_SimpleNew(1, phi_max_dim, NPY_INT);
  PyArrayObject *py_pix_q_array = 
    (PyArrayObject *) PyArray_SimpleNew(1, phi_max_dim, NPY_INT);
  if (!py_dtde_q || !py_q_pix_array || !py_pix_q_array) {
    return NULL;
  }
 
  /* put arguments into variables */
  if (!PyArg_ParseTuple(args, "Od", &opy_dtde_l, &cte_frac)) {
    return NULL;
  }
   
  py_dtde_l = (PyArrayObject *) PyArray_FROMANY(opy_dtde_l, NPY_DOUBLE, 1, 1, NPY_IN_ARRAY);
  if (!py_dtde_l) {
    return NULL;
  }
  
  /* copy input python array to local C array */
  for (p = 0; p < NUM_PHI; p++) {
    dtde_l[p] = *(double *) PyArray_GETPTR1(py_dtde_l, p);
  }
  
  /* call InterpolatePhi */
  status = InterpolatePhi(dtde_l, cte_frac, dtde_q, q_pix_array, pix_q_array, &ycte_qmax);
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in InterpolatePhi.");
    return NULL;
  }
  
  /* copy local C arrays to return variables */
  for (p = 0; p < MAX_PHI; p++) {
    *(double *) PyArray_GETPTR1(py_dtde_q, p) = dtde_q[p];
    *(int *) PyArray_GETPTR1(py_q_pix_array, p) = q_pix_array[p];
    *(int *) PyArray_GETPTR1(py_pix_q_array, p) = pix_q_array[p];
  }
  
  Py_DECREF(py_dtde_l);
  
  return Py_BuildValue("OOOi",py_dtde_q, py_q_pix_array, py_pix_q_array, ycte_qmax);
}

static PyObject * py_TrackChargeTrap(PyObject *self, PyObject *args) {
  /* input variables */
  PyObject *opy_pix_q_array, *opy_chg_leak;
  PyArrayObject *py_pix_q_array, *py_chg_leak;
  int ycte_qmax;
  
  /* put arguments into variables */
  if (!PyArg_ParseTuple(args, "OOi", &opy_pix_q_array, &opy_chg_leak, &ycte_qmax)) {
    return NULL;
  }
  
  py_pix_q_array = (PyArrayObject *) PyArray_FROMANY(opy_pix_q_array, NPY_INT, 1, 1, NPY_IN_ARRAY);
  py_chg_leak = (PyArrayObject *) PyArray_FROMANY(opy_chg_leak, NPY_DOUBLE, 2, 2, NPY_IN_ARRAY);
  if (!py_pix_q_array || !py_chg_leak) {
    return NULL;
  }
  
  /* local variables */
  int status, i, j;
  int pix_q_array[MAX_PHI];
  double chg_leak[MAX_TAIL_LEN*NUM_LOGQ];
  double chg_leak_tq[MAX_TAIL_LEN*ycte_qmax];
  double chg_open_tq[MAX_TAIL_LEN*ycte_qmax];
  
  /* return variables */
  npy_intp out_dim[] = {MAX_TAIL_LEN, ycte_qmax};
  PyArrayObject *py_chg_leak_tq = 
    (PyArrayObject *) PyArray_SimpleNew(2, out_dim, NPY_DOUBLE);
  PyArrayObject *py_chg_open_tq = 
    (PyArrayObject *) PyArray_SimpleNew(2, out_dim, NPY_DOUBLE);
  if (!py_chg_leak_tq || !py_chg_open_tq) {
    return NULL;
  }
  
  /* copy input python arrays to local C arrays */
  for (i = 0; i < MAX_PHI; i++) {
    pix_q_array[i] = *(int *) PyArray_GETPTR1(py_pix_q_array, i);
  }
  
  for (i = 0; i < MAX_TAIL_LEN; i++) {
    for (j = 0; j < NUM_LOGQ; j++) {
      chg_leak[i*NUM_LOGQ + j] = *(double *) PyArray_GETPTR2(py_chg_leak, i, j);
    }
  }
  
  /* call TrackChargeTrap */
  status  = TrackChargeTrap(pix_q_array, chg_leak, ycte_qmax, chg_leak_tq, chg_open_tq);
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in TrackChargeTrap.");
    return NULL;
  }
  
  /* copy local C arrays to return variables */
  for (i = 0; i < MAX_TAIL_LEN; i++) {
    for (j = 0; j < ycte_qmax; j++) {
      *(double *) PyArray_GETPTR2(py_chg_leak_tq, i, j) = chg_leak_tq[i*ycte_qmax + j];
      *(double *) PyArray_GETPTR2(py_chg_open_tq, i, j) = chg_open_tq[i*ycte_qmax + j];
    }
  }
  
  Py_DECREF(py_pix_q_array);
  Py_DECREF(py_chg_leak);
  
  return Py_BuildValue("OO",py_chg_leak_tq, py_chg_open_tq);
}

static PyObject * py_DecomposeRN(PyObject *self, PyObject *args) {
  /* input variables */
  PyObject *opy_data;
  PyArrayObject *py_data;
  int model, nitr;
  double readnoise;
  
  /* put arguments into variables */
  if (!PyArg_ParseTuple(args, "Oiid", &opy_data, &model, &nitr, &readnoise)) {
    return NULL;
  }
  
  py_data = (PyArrayObject *) PyArray_FROMANY(opy_data, NPY_DOUBLE, 2, 2, NPY_IN_ARRAY);
  if (!py_data) {
    return NULL;
  }
  
  /* local variables */
  int status, i, j;
  int arrx = py_data->dimensions[0];
  int arry = py_data->dimensions[1];
  double * data;
  double * sig;
  double * noise;
  data = (double *) malloc(arrx * arry * sizeof(double));
  sig = (double *) malloc(arrx * arry * sizeof(double));
  noise = (double *) malloc(arrx * arry * sizeof(double));
  
  /* return variables */
  npy_intp out_dim[] = {arrx, arry};
  PyArrayObject *py_sig = (PyArrayObject *) PyArray_SimpleNew(2, out_dim, NPY_DOUBLE);
  PyArrayObject *py_noise = (PyArrayObject *) PyArray_SimpleNew(2, out_dim, NPY_DOUBLE);
  if (!py_sig || !py_noise) {
    return NULL;
  }
  
  /* copy input Python arrays to local C arrays */
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      data[i*arry + j] = *(double *) PyArray_GETPTR2(py_data, i, j);
    }
  }
  
  /* call DecomposeRN */
  status = DecomposeRN(arrx, arry, data, model, nitr, readnoise, sig, noise);
  if (status != 0) {
    PyErr_SetString(PyExc_StandardError, "An error occurred in DecomposeRN.");
    return NULL;
  }
  
  /* copy local C arrays to return Python arrays */
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      *(double *) PyArray_GETPTR2(py_sig, i, j) = sig[i*arry + j];
      *(double *) PyArray_GETPTR2(py_noise, i, j) = noise[i*arry + j];
    }
  }
  
  free(data);
  free(sig);
  free(noise);
  
  Py_DECREF(py_data);
  
  return Py_BuildValue("OO", py_sig, py_noise);
}

static PyObject * py_FixYCte(PyObject *self, PyObject *args) {
  /* input variables */
  PyObject *opy_sig_cte, *opy_q_pix, *opy_chg_leak, *opy_chg_open;
  PyArrayObject *py_sig_cte, *py_q_pix, *py_chg_leak, *py_chg_open;
  int ycte_qmax;
  const char *amp_name, *log_file;
  
  /* put arguments into variables */
  if (!PyArg_ParseTuple(args, "OiOOOss", &opy_sig_cte, &ycte_qmax, &opy_q_pix, 
                        &opy_chg_leak, &opy_chg_open, &amp_name, &log_file)) {
    return NULL;
  }
  
  py_sig_cte = (PyArrayObject *) PyArray_FROMANY(opy_sig_cte, NPY_DOUBLE, 2, 2, NPY_IN_ARRAY);
  py_q_pix = (PyArrayObject *) PyArray_FROMANY(opy_q_pix, NPY_INT, 1, 1, NPY_IN_ARRAY);
  py_chg_leak = (PyArrayObject *) PyArray_FROMANY(opy_chg_leak, NPY_DOUBLE, 2, 2, NPY_IN_ARRAY);
  py_chg_open = (PyArrayObject *) PyArray_FROMANY(opy_chg_open, NPY_DOUBLE, 2, 2, NPY_IN_ARRAY);
  if (!py_sig_cte || !py_q_pix || !py_chg_leak || !py_chg_open) {
    return NULL;
  }
  
  /* local variables */
  int status, i, j;
  int q_pix[MAX_PHI];
  int arrx = py_sig_cte->dimensions[0];
  int arry = py_sig_cte->dimensions[1];
  double * sig_cte;
  double * sig_cor;
  double * chg_leak;
  double * chg_open;
  sig_cte = (double *) malloc(arrx * arry * sizeof(double));
  sig_cor = (double *) malloc(arrx * arry * sizeof(double));
  chg_leak = (double *) malloc(MAX_TAIL_LEN * ycte_qmax * sizeof(double));
  chg_open = (double *) malloc(MAX_TAIL_LEN * ycte_qmax * sizeof(double));
  
  /* return variables */
  npy_intp out_dim[] = {arrx, arry};
  PyArrayObject *py_sig_cor = (PyArrayObject *) PyArray_SimpleNew(2, out_dim, NPY_DOUBLE);
  if (!py_sig_cor) {
    return NULL;
  }
  
  /* copy input Python arrays to local C arrays */
  for (i = 0; i < MAX_PHI; i++) {
    q_pix[i] = *(int *) PyArray_GETPTR1(py_q_pix, i);
  }
  
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      sig_cte[i*arry + j] = *(double *) PyArray_GETPTR2(py_sig_cte, i, j);
    }
  }
  
  for (i = 0; i < MAX_TAIL_LEN; i++) {
    for (j = 0; j < ycte_qmax; j++) {
      chg_leak[i*ycte_qmax + j] = *(double *) PyArray_GETPTR2(py_chg_leak, i, j);
      chg_open[i*ycte_qmax + j] = *(double *) PyArray_GETPTR2(py_chg_open, i, j);
    }
  }
  
  /* call FixYCte */
  status = FixYCte(arrx, arry, sig_cte, sig_cor, ycte_qmax, q_pix,
                   chg_leak, chg_open, amp_name, log_file);
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
  
  free(sig_cte);
  free(sig_cor);
  free(chg_leak);
  free(chg_open);
  
  Py_DECREF(py_sig_cte);
  Py_DECREF(py_q_pix);
  Py_DECREF(py_chg_leak);
  Py_DECREF(py_chg_open);
  
  return Py_BuildValue("O", py_sig_cor);
}

static PyMethodDef PixCte_FixY_methods[] =
{
  {"CalcCteFrac",  py_CalcCteFrac, METH_VARARGS, "Calculate CTE multiplier."},
  {"InterpolatePsi", py_InterpolatePsi, METH_VARARGS, "Interpolate to full tail profile."},
  {"InterpolatePhi", py_InterpolatePhi, METH_VARARGS, "Interpolate charge traps."},
  {"TrackChargeTrap", py_TrackChargeTrap, METH_VARARGS, "TrackChargeTrap."},
  {"DecomposeRN", py_DecomposeRN, METH_VARARGS, "Separate signal and readnoise."},
  {"FixYCte", py_FixYCte, METH_VARARGS, "Perform parallel CTE correction."},
  {NULL, NULL, 0, NULL} /* sentinel */
  
};

PyMODINIT_FUNC initPixCte_FixY(void)
{
  (void) Py_InitModule("PixCte_FixY", PixCte_FixY_methods);
  import_array(); // Must be present for NumPy
}
