/*
    Program: PixCte_FixY.c
    Authors: P. L. Lim (C), J. Anderson (Fortran)
    Purpose: Perform computationally intensive operations for 
             pixel-based CTE correction in parallel direction.

    References:
    http://docs.scipy.org/doc/numpy/reference/c-api.html
    http://www.scipy.org/Cookbook/C_Extensions/NumPy_arrays
    http://dsnra.jpl.nasa.gov/software/Python/numpydoc/numpy-13.html

    History:
    2010-09-16 Created by PLL.
*/

#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <math.h>
#include <string.h>

/* Global constants - ACS/WFC only */
#define MAX_YSIZE 2048 // Full-frame FLT - do not change
#define J_PRINT_MIN 1996 // 1997-1
#define J_PRINT_MAX 2015 // 2015-0
#define TRPQ_INIT 99 // 100-1

/* Global constants */
#define ENDLINE "\n"
#define N_ITER 4     // N interations for sim_readout()

/* Global constants - NOT USED. Kept just in case. */
//#define PMAX 49998   // 49999-1
//#define QUMAX 9999   // 10000-1

/* Global variables */
int QMAX = 99999;    // Will be updated in FixYCte()

/* #######################################################
   Matrix Utility functions
   http://www.scipy.org/Cookbook/C_Extensions/NumPy_arrays */

/* Allocate a double *vector (vec of pointers). */
double ** ptrvector(long n)
{
    double **v;
    v = (double **) malloc((size_t) (n*sizeof(double)));
    if (!v)
    {
        printf("In **ptrvector. Allocation of memory for double array failed.");
        exit(0);
    }
    return v;
}

/* Free a double *vector (vec of pointers). */ 
void free_Carrayptrs(double **v)
{
    free((char*) v);
}

/* Create Carray from PyArray. Assumes PyArray is contiguous in memory. */
double ** pymatrix_to_Carrayptrs(PyArrayObject *arrayin)
{
    double **c, *a;
    int i, n, m;
    n = arrayin->dimensions[0];
    m = arrayin->dimensions[1];
    c = ptrvector(n);
    a = (double *) arrayin->data; /* pointer to arrayin data as double */
    for (i=0; i<n; i++) c[i] = a + i*m;
    return c;
}

/* ####################################################### */

/* Evaluate chg_leak_tq or chg_open_tq */
double eval_prof(PyArrayObject *chg_tq, int t, int q, double val4tmax)
{
    double rval=0.0;
    int qu=q, qu_max=chg_tq->dimensions[1]-1, t_max=chg_tq->dimensions[0]-1;

    if (t < 0) return rval; // In Fortran, rval=0 if t=0
    if (t > t_max) return val4tmax;
    if (q < 0) return rval;

    if (q > qu_max) qu = qu_max;
    rval = *(double *)(chg_tq->data + t*chg_tq->strides[0] + qu*chg_tq->strides[1]);
    return rval;
}

/* Evaluate q_pix_array */
int eval_qpix(PyArrayObject *q_pix_array, double p)
{
    double rval=0.0;
    int ip, ip_max=q_pix_array->dimensions[0]-1;

    if (p < 1) return (int) rval;

    ip = ((int) p) - 1;
    if (ip > ip_max) ip = ip_max;
    rval = *(double *)(q_pix_array->data + ip*q_pix_array->strides[0]);
    return (int) rval;
}

/*
This version will use the arrays that go from 0 to QMAX-1
to describe the CTE loss.

INPUTS:
pix_curr: The "true" array of pixel values.
y_size  : Number of rows for readout in each column.
q_pix_array: Calculated in Python code.
chg_leak_tq: Calculated in Python code.
chg_open_tq: Calculated in Python code.

OUTPUT:
pix_read: The "observed" array, resulting from CTE + "true"
*/
void sim_readout(double pix_curr[], double pix_read[], int y_size, PyArrayObject *q_pix_array, PyArrayObject *chg_leak_tq, PyArrayObject *chg_open_tq)
{
    int i, q, t, trpq[QMAX], trpq_init2=TRPQ_INIT+1;
    double iy, qpix, free1, fill;

    // This never needs to be larger than QMAX
    for (q=0;q<QMAX;q++) trpq[q] = trpq_init2;

    /*
    Go through the readout cycle. Read the end pixel first
    and put it next onto the buffer. Then shift the others
    over. Do this y_size times.
    */
    for (i=0;i<y_size;i++)
    {
        iy = (i + 1.0) / MAX_YSIZE;

        /*
        This first half of the loop deals with the emptying
        of traps. trpq[q] tells you how long it has been
        since trap #q has released its charge.
        */
        free1 = 0.0;
        q = 0;
        while (q<QMAX && trpq[q]<trpq_init2)
	{
            t = trpq[q] + 1;
	    if (t <= trpq_init2)
	    {
	      /*
	      Once a trap has given off its electrons for
              this shift, then increment how long it's been
	      since it was filled.
	      */
	      free1 += eval_prof(chg_leak_tq, t-1, q, 0.0) * iy;
              trpq[q]++;
	    }
  	    q++;
        } // End of q loop of emptying traps

	pix_read[i] = pix_curr[i] + free1;

	/*
	Second half of the loop deals with the filling of
	traps. It is easier to confine fewer electrons.
	A larger electron cloud sees more traps as a cloud
	but less traps per electron. Trap density is
	probably uniform but electron cloud is not.
	*/
	fill = 0.0;
	qpix = eval_qpix(q_pix_array, pix_read[i]);
	for (q=0;q<qpix;q++)
	{
	    t = trpq[q];
	    fill += eval_prof(chg_open_tq, t-1, q, 1.0) * iy;
	    trpq[q] = 0;
	} // End of q loop of filling traps

	pix_read[i] -= fill;
    } // End of i loop through a column
}

/*
Like sim_readout() but slightly different implementation.
Kept for testing only. 
*/
void sim_readout_NOT_USED(double pix_curr[], double pix_read[], int y_size, PyArrayObject *q_pix_array, PyArrayObject *chg_leak_tq, PyArrayObject *chg_open_tq)
{
    int i, q, t, trpq[QMAX], trpq_init2=TRPQ_INIT+1;
    double iy, qpix, free1, fill;

    // This never needs to be larger than QMAX
    for (q=0;q<QMAX;q++) trpq[q] = trpq_init2;

    /*
    Go through the readout cycle. Read the end pixel first
    and put it next onto the buffer. Then shift the others
    over. Do this y_size times.
    */
    for (i=0;i<y_size;i++)
    {
        iy = (i + 1.0) / MAX_YSIZE;
        free1 = 0.0;

	/* ORIGINAL LOOP STYLE
	for (q=0;q<QMAX;q++)
	{
	    if (trpq[q] >= trpq_init2) break;
	    t = trpq[q] + 1;
	    if (t <= trpq_init2)
	    {
	        free1 += eval_prof(chg_leak_tq, t-1, q, 0.0) * iy;
		trpq[q]++;
	    }
	}
	*/

	/* IMPROVED LOOP STYLE */
	q = 0;
        while (q<QMAX && trpq[q]<trpq_init2)
	{
            t = trpq[q] + 1;
	    if (t <= trpq_init2)
	    {
	      free1 += eval_prof(chg_leak_tq, t-1, q, 0.0) * iy;
              trpq[q]++;
	    }
  	    q++;
        }

	pix_read[i] = pix_curr[i] + free1;

	fill = 0.0;
	qpix = eval_qpix(q_pix_array, pix_read[i]);
	for (q=1;q<=qpix;q++)
	{
	    t = trpq[q-1];
	    fill += eval_prof(chg_open_tq, t-1, q-1, 1.0) * iy;
            trpq[q-1] = 0;
	}

	pix_read[i] -= fill;
    } // End of i loop through a column
}

static PyObject * FixYCte(PyObject *self, PyObject *args)
{
    // Declaration and initialization
    PyObject *oimage, *ooutimage, *oqpixarr, *ochgleak, *ochgopen;
    PyArrayObject *image, *outimage, *qpixarr, *chgleak, *chgopen;
    int i, j, n, xSize, ySize, doLog=0;
    double in_qmax, pix_obsd[MAX_YSIZE], pix_curr[MAX_YSIZE], pix_read[MAX_YSIZE], **cout;
    const char *amp_name, *log_file;
    FILE *flog;

    /*
    1. Quadrant image
    2. Output image
    3. QMAX = _YCTE_QMAX
    4. q_pix_array
    5. chg_leak_tq
    6. chg_open_tq
    7. AMP
    8. Log file name

    FUTURE WORK: Might also need to read in NITS from Python.
    */
    if (!PyArg_ParseTuple(args, "OOdOOOss:FixYCte", &oimage, &ooutimage, &in_qmax, &oqpixarr, &ochgleak, &ochgopen, &amp_name, &log_file)) return NULL;

    // Build arrays
    image = (PyArrayObject *)PyArray_FROM_OTF(oimage, NPY_DOUBLE, NPY_IN_ARRAY);
    outimage = (PyArrayObject *)PyArray_FROM_OTF(ooutimage, NPY_DOUBLE, NPY_OUT_ARRAY);
    qpixarr = (PyArrayObject *)PyArray_FROM_OTF(oqpixarr, NPY_DOUBLE, NPY_IN_ARRAY);
    chgleak = (PyArrayObject *)PyArray_FROM_OTF(ochgleak, NPY_DOUBLE, NPY_IN_ARRAY);
    chgopen = (PyArrayObject *)PyArray_FROM_OTF(ochgopen, NPY_DOUBLE, NPY_IN_ARRAY);
    if (!image || !outimage || !qpixarr || !chgleak || !chgopen) return NULL;

    // Image dimensions
    xSize = image->dimensions[1];
    ySize = image->dimensions[0];

    // Change output to C - memory is allocated
    cout = pymatrix_to_Carrayptrs(outimage);

    // Set global QMAX (comment out to use preset value)
    QMAX = (int) in_qmax;

    // Open optional log file
    if (strlen(log_file) > 0)
    {
        doLog = 1;
        flog = fopen(log_file, "a"); // Always append

	// Header
        fprintf(flog, "#%s# AMP: %s%s", ENDLINE, amp_name, ENDLINE);
	fprintf(flog, "#%-4s ", "#PIX");
	for (j=J_PRINT_MIN; j<J_PRINT_MAX; j++) fprintf(flog, "Y=%-4i ", j+1);
	fprintf(flog, "%s", ENDLINE);
    }

    // Loop through columns. Columns are independent of each other.
    for (i=0; i<xSize; i++)
    {
        // Copy input column
        for (j=0; j<ySize; j++)
	{
 	    pix_obsd[j] = *(double *)(image->data + i*image->strides[1] + j*image->strides[0]);
	    pix_curr[j] = pix_obsd[j];
	    pix_read[j] = 0.0;
	}

	/*
	Pixel-based CTE correction.

	FUTURE WORK: NITS will be the number of iteration of
	an extra loop somwhere here to call sim_readout
	multiple times to refine qpixarr(?).
	*/
	for (n=0; n<N_ITER; n++)
	{
	  sim_readout(pix_curr, pix_read, ySize, qpixarr, chgleak, chgopen);
	  for (j=0; j<ySize; j++) pix_curr[j] += pix_obsd[j] - pix_read[j];
	}

	// Copy output column
	for (j=0; j<ySize; j++) cout[j][i] = pix_curr[j];

	// Write to log file
	if (doLog)
        {
	    fprintf(flog, "# X=%-i%s", i+1, ENDLINE);

  	    // PIX_CURR
	    fprintf(flog, "%-4s ", "CURR");
            for (j=J_PRINT_MIN; j<J_PRINT_MAX; j++) fprintf(flog, "%6i ", (int)(pix_curr[j]+0.5));
	    fprintf(flog, "%s", ENDLINE);

	    // PIX_OBSD
	    fprintf(flog, "%-4s ", "OBSD");
            for (j=J_PRINT_MIN; j<J_PRINT_MAX; j++) fprintf(flog, "%6i ", (int)(pix_obsd[j]+0.5));
	    fprintf(flog, "%s", ENDLINE);

	    // PIX_READ
	    fprintf(flog, "%-4s ", "READ");
            for (j=J_PRINT_MIN; j<J_PRINT_MAX; j++) fprintf(flog, "%6i ", (int)(pix_read[j]+0.5));
	    fprintf(flog, "%s", ENDLINE);
	}
    }

    // Close optional log file
    if (doLog) fclose(flog);

    // Destroy array references
    Py_XDECREF(image);
    Py_XDECREF(qpixarr);
    Py_XDECREF(chgleak);
    Py_XDECREF(chgopen);

    /*Py_XINCREF(outimage); */
    free_Carrayptrs(cout);

    return Py_BuildValue("i",0);
}

static PyMethodDef PixCte_FixY_methods[] =
{
    {"FixYCte",  FixYCte, METH_VARARGS, "Fix parallel CTE"},
    {NULL, NULL, 0, NULL} /* sentinel */

};

void initPixCte_FixY(void)
{
    Py_InitModule("PixCte_FixY", PixCte_FixY_methods);
    import_array(); // Must be present for NumPy
}
