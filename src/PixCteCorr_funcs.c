#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "PixCteCorr.h"

#define ERROR_RETURN 2

/*
 * CalcCteFrac calculates the multiplicative factor that accounts for the
 * worsening of CTE over time. This is currently a linear function valid over
 * the whole life of ACS/WFC, but this will change soon. Jay Anderson has
 * discovered that the slope of the CTE scaling is not constant so the function
 * for CTE frac will be different depending on the obs. start time. The plan
 * is to add the CTE frac parameterization to the CTE params reference file
 * once Jay has them nailed down.
 * - MRD 18 Feb. 2011
 *
 * Constants for instrument names are defined in PixCteCorr.h.
 */
double CalcCteFrac(const double mjd, const int instrument) {
  
  /* variables used to calculate the CTE scaling slope */
  double mjd_pt1, mjd_pt2; /* the MJD points at which the scaling is defined */
  double cte_pt1, cte_pt2; /* the CTE frac points at which the scaling is defined */
  
  double cte_frac;         /* return value */
  
  if (instrument == ACSWFC) {
    cte_pt1 = 0.0;
    cte_pt2 = 1.0;
    mjd_pt1 = 52335.0;  /* March 2, 2002 */
    mjd_pt2 = 55263.0;  /* March 8, 2010 */
  } else {
    printf("Instrument not found: %i\n",instrument);
    return -9999.0;
  }
  
  cte_frac = ((cte_pt2 - cte_pt1) / (mjd_pt2 - mjd_pt1)) * (mjd - mjd_pt1);
  
  return cte_frac;
}

/*
 * InterpolatePsi fills in the sparser array describing trail profiles read
 * from the CTE parameters file. Is there any reason the whole profile can't
 * be written to the parameters file? It's only 100 elements long.
 * - MRD 18 Feb. 2011
 *
 * Inputs chg_leak and psi_node are arrays read from the CTE parameters file.
 * Output chg_leak_interp has all the data chg_leak plus interpolated data where
 * chg_leak has none.
 */
int InterpolatePsi(const double chg_leak[NUM_PSI*NUM_LOGQ], const int psi_node[NUM_PSI],
                   double chg_leak_interp[MAX_TAIL_LEN*NUM_LOGQ],
                   double chg_open_interp[MAX_TAIL_LEN*NUM_LOGQ]) {
  
  /* status variable for return */
  int status = 0;
  
  /* index variables for tracking where we are in psi_node/chg_leak.
   * these will always be one apart. so we don't really need two, but
   * I like it for cleanliness. */
  int pn_i1 = 0;
  int pn_i2 = 1;
  
  double interp_frac; /* the fraction of the distance between psi_node1 and
                       * psi_node2 are we interpolating at */
  
  double sum_rel;     /* total probability of release */
  double sum_cum;     /* running total probability of release */
  
  /* iteration variables */
  int i, j;
  
  /* loop over all pixels in the trail and calculate the profile at each q
   * if it isn't already in chg_leak */
  for (i = 0; i < MAX_TAIL_LEN; i++) { 
    /* do we match an existing chg_leak row? */
    if (i+1 == psi_node[pn_i1]) {
      /* if so, copy it over */
      for (j = 0; j < NUM_LOGQ; j++) {
        chg_leak_interp[i*NUM_LOGQ + j] = chg_leak[pn_i1*NUM_LOGQ + j];
      }      
    } else {
      /* no match, need to interpolate */
      interp_frac = ((double) (i+1 - psi_node[pn_i1])) / 
                    ((double) (psi_node[pn_i2] - psi_node[pn_i1]));
      /* loop over each q column */
      for (j = 0; j < NUM_LOGQ; j++) {
        chg_leak_interp[i*NUM_LOGQ + j] = chg_leak[pn_i1*NUM_LOGQ + j] + 
        (interp_frac * 
         (chg_leak[pn_i2*NUM_LOGQ + j] - chg_leak[pn_i1*NUM_LOGQ + j]));
      }
    }
    /* if the next of row psi_node is an existing row in chg_leak we should
     * increment our indices so we start interpolating between the next pair */
    if ((i+2 == psi_node[pn_i2]) && (i < MAX_TAIL_LEN)) {
      pn_i1++;
      pn_i2++;
    }
  }
  
  /* perform tail normalization and cumulative release probability calculation */
  for (i = 0; i < NUM_LOGQ; i++) {
    sum_rel = 0.0;
    
    /* get total in this Q column */
    for (j = 0; j < MAX_TAIL_LEN; j++) {
      sum_rel += chg_leak_interp[j*NUM_LOGQ + i];
    }
    
    /* normalize chg_leak_interp by total */
    for (j = 0; j < MAX_TAIL_LEN; j++) {
      chg_leak_interp[j*NUM_LOGQ + i] = chg_leak_interp[j*NUM_LOGQ + i]/sum_rel;
    }
    
    /* calculate cumulative probability of release */
    sum_cum = 0.0;
    
    for (j = 0; j < MAX_TAIL_LEN; j++) {
      sum_cum += chg_leak_interp[j*NUM_LOGQ + i];
      chg_open_interp[j*NUM_LOGQ + i] = 1.0 - sum_cum;
    }
  }
  
  return status;
}

/*
 * InterpolatePhi interpolates information read from the CTE parameters file
 * that describes the amount of charge in the CTE tail and where the charge is.
 * -MRD 21 Feb. 2011
 *
 * Input dtde_l is read from the CTE parameters file and cte_frac is calculated
 * from the observation start date by CalcCteFrac.
 * Outputs dtde_q, q_pix_array, and pix_q_array are arrays MAX_PHI long (should
 * be 99999) and ycte_qmax is an integer.
 */
int InterpolatePhi(const double dtde_l[NUM_PHI], const int q_dtde[NUM_PHI],
                   const int shft_nit, double dtde_q[MAX_PHI]) {
  
  /* status variable for return */
  int status = 0;
  
  int p; /* iteration variable over phi nodes in reference file */
  int q; /* iteration variable over single phi values between nodes in ref file */
  
  /* interpolation calculation variables */
  double interp_pt;   /* point at which we're interpolating data */
  double interp_dist; /* difference between interp_pt and low_node */
  double interp_val;  /* interpolated value */
  
  /* upper and lower bounds of interpolation range */
  double log_qa, log_qb;
  double log_da, log_db;
  
  /* something for holding intermediate calculation results */
  double qtmp;
  
  for (p = 0; p < NUM_PHI-1; p++) {
    log_qa = log10((double) q_dtde[p]);
    log_qb = log10((double) q_dtde[p+1]);
    log_da = log10(dtde_l[p]);
    log_db = log10(dtde_l[p+1]);
    
    for (q = q_dtde[p]; q < q_dtde[p+1]; q++) {
      interp_pt = log10((double) q);
      interp_dist = (interp_pt - log_qa) / (log_qb - log_qa);
      interp_val = log_da + (interp_dist * (log_db - log_da));
      
      qtmp = pow(10, interp_val)/(double) CTE_REF_ROW;
      qtmp = pow((1.0 - qtmp), (double) CTE_REF_ROW/ (double) shft_nit);
      dtde_q[q-1] = 1.0 - qtmp;
    }
  }
  
  qtmp = pow((1.0 - (dtde_l[NUM_PHI-1]/CTE_REF_ROW)),CTE_REF_ROW/shft_nit);
  dtde_q[MAX_PHI-1] = 1.0 - qtmp;
  
  return status;
}

/* In this function we're interpolating the tail arrays over the Q dimension
 * and reducing the arrays to contain data at only the charge levels
 * specified in the levels array. */
int FillLevelArrays(const double chg_leak_kt[MAX_TAIL_LEN*NUM_LOGQ],
                    const double chg_open_kt[MAX_TAIL_LEN*NUM_LOGQ],
                    const double dtde_q[MAX_PHI], const int levels[NUM_LEV],
                    double chg_leak_lt[MAX_TAIL_LEN*NUM_LEV],
                    double chg_open_lt[MAX_TAIL_LEN*NUM_LEV],
                    double dpde_l[NUM_LEV],
                    int tail_len[NUM_LEV]) {
  
  /* status variable for return */
  int status = 0;
  
  int l,t;  /* iteration variables for tail and levels */
  int q;    /* iteration variable for q levels in between those specified in levels */
  
  /* container for cumulative dtde_q */
  double cpde_l[NUM_LEV];
  
  /* variable for running sum of dtde_q */
  double sum = 0.0;
  
  int logq_ind; /* index of lower logq used for interpolation */
  
  double logq;           /* log of charge level */
  double logq_min = 1.0; /* min value for logq */
  double logq_max = 3.999; /* max value for logq */
  
  double interp_dist; /* difference between logp and lower logq */
  
  dpde_l[0] = 0.0;
  cpde_l[0] = 0.0;
  
  for (t = 0; t < MAX_TAIL_LEN; t++) {
    chg_leak_lt[t*NUM_LEV] = chg_leak_kt[t*NUM_LOGQ];
    chg_open_lt[t*NUM_LEV] = chg_open_kt[t*NUM_LOGQ];
  }
  
  for (l = 1; l < NUM_LEV; l++) {
    for (q = levels[l-1]; q < levels[l]; q++) {
      sum += dtde_q[q];
    }
    
    cpde_l[l] = sum;
    dpde_l[l] = cpde_l[l] - cpde_l[l-1];
    
    /* calculate logq with min/max clipping */
    logq = log10((double) q);
    if (logq < logq_min) {
      logq = logq_min;
    } else if (logq > logq_max) {
      logq = logq_max;
    }
    
    /* set logq_ind for this logq */
    if (logq < 2) {
      logq_ind = 0;
    } else if (logq < 3) {
      logq_ind = 1;
    } else {
      logq_ind = 2;
    }
    
    interp_dist = logq - floor(logq);
    
    for (t = 0; t < MAX_TAIL_LEN; t++) {
      chg_leak_lt[t*NUM_LEV + l] = ((1.0 - interp_dist) * chg_leak_kt[t*NUM_LOGQ + logq_ind]) +
                                   (interp_dist * chg_leak_kt[t*NUM_LOGQ + logq_ind+1]);
      chg_open_lt[t*NUM_LEV + l] = ((1.0 - interp_dist) * chg_open_kt[t*NUM_LOGQ + logq_ind]) +
                                   (interp_dist * chg_open_kt[t*NUM_LOGQ + logq_ind+1]);
    }
  }
  
  /* calculate max tail lengths for each level */
  for (l = 0; l < NUM_LEV; l++) {
    tail_len[l] = MAX_TAIL_LEN;
    
    for (t = MAX_TAIL_LEN-1; t >= 0; t--) {
      if (chg_leak_lt[t*NUM_LEV + l] == 0) {
        tail_len[l] = t+1;
      } else {
        break;
      }
    }
  }
  
  return status;
}

/*
 * Attempt to separate readout noise from signal, since CTI happens before
 * readout noise is added to the signal.
 *
 * The model parameter can be 0, 1, or 100:
 *   0: no separation
 *   1: linear smoothing along columns (this should be the default)
 *   100: version of 1 that uses the reported readout noise and no iteration
 *
 * The nitr parameters controls the number of iterations of smoothing applied in
 * in model #1. It is one of the parameters read from the CTE parameters file.
 * 
 * The readnoise parameter is the readout noise from the CCDTAB for this amp
 * and is only used in model 100.
 */
int DecomposeRN(const int arrx, const int arry, const double data[arrx*arry], 
                const int model, const int nitr, const double readnoise,
                double sig_arr[arrx*arry], double noise_arr[arrx*arry]) {
  
  /* status variable for return */
  int status = 0;
  
  /* iteration variables */
  int i, j;
  
  /* copy the science data to the signal array and make sure noise array is 0*/
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      sig_arr[i*arry + j] = data[i*arry + j];
      noise_arr[i*arry + j] = 0.0;
    }
  }
  
  if (model == 1) {
    status = DecomposeRNModel1(arrx, arry, data, nitr, sig_arr, noise_arr);
  } else if (model == 100) {
    status = DecomposeRNModel100(arrx, arry, readnoise, data, sig_arr, noise_arr);
  }
  
  return status;
}

/*
 * Separate the readout noise and signal using smoothing along the columns.
 *
 * nitr controls the number of iterations of smoothing.
 */
int DecomposeRNModel1(const int arrx, const int arry, 
                      const double data[arrx*arry], const int nitr,
                      double sig[arrx*arry], double noise[arrx*arry]) {
  
  /* status variable for return */
  int status = 0;
  
  /* iteration variables */
  int i, j, n;
  
  /* only do corrections between these thresholds */
  int noiseLo = -20;
  int noiseHi = 100;
  
  /* variable to hold residuals */
  double res;
  
  double * sig_temp;
  
  int * doNsCor;
  
  /* make a temporary array to hold the signal while doing the smoothing
   * calculations. this way we can keep the rows in their old values while
   * doing calculations on adjacent rows. */
  sig_temp = (double *) calloc(arrx * arry, sizeof(double));
  
  /* set up array of flags for which pixels should be noise corrected */
  doNsCor = (int *) malloc(arrx * arry * sizeof(int));
  
  /* let's start with all ones */
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      doNsCor[i*arry + j] = 1;
    }
  }
  
  /* then turn off any pixels we shouldn't change */
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      if (i == 0 || i == arrx-1) {
        doNsCor[i*arry + j] = 0;
      } else if (data[i*arry + j] < noiseLo || data[i*arry + j] > noiseHi) {
        doNsCor[i*arry + j] = 0;
        doNsCor[(i-1)*arry + j] = 0;
        doNsCor[(i+1)*arry + j] = 0;
      }
    }
  }
  
  /* loop over number of iterations */
  for (n = 0; n < nitr; n++) {
    for (i = 1; i < arrx-1; i++) {
      for (j = 0; j < arry; j++) {
        if (doNsCor[i*arry + j] == 1) {
          
          /* compute residual as center pixel - mean of neighbors */
          res = sig[i*arry + j] - (0.5 * (sig[(i-1)*arry + j] + sig[(i+1)*arry + j]));
          
          /* clip residual to +/- 1 electron */
          if (res < -1) {
            res = -1;
          } else if (res > 1) {
            res = 1;
          }
          
          /* subtract residual from signal */
          sig_temp[i*arry + j] = sig[i*arry + j] - res;
        }
      }
    }
    
    /* now that we're done with this iteration copy sig_temp back to sig */
    for (i = 1; i < arrx-1; i++) {
      for (j = 0; j < arry; j++) {
        if (doNsCor[i*arry + j] == 1) {
          sig[i*arry + j] = sig_temp[i*arry + j];
        }
      }
    }
  }
  
  free(doNsCor);
  free(sig_temp);
  
  /* compute the readnoise as the difference between the original data and
   * the corrected data. */
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      noise[i*arry + j] = data[i*arry + j] - sig[i*arry + j];
    }
  }
  
  return status;
}

/*
 * Separate readout noise from signal using the READNSE parameter from CCDTAB
 * and no iterative smoothing.
 */
int DecomposeRNModel100(const int arrx, const int arry, const double readnoise,
                        const double data[arrx*arry],
                        double sig[arrx*arry], double noise[arrx*arry]) {
  
  /* status variable for return */
  int status = 0;
  
  /* iteration variables */
  int i, j;
  
  /* only do corrections between these thresholds */
  double sigCut = 2;
  double noiseLo = sigCut * readnoise;
  double noiseHi = sigCut * noiseLo;
  
  /* exclude pixels on the near and far sides relative to the amp */
  int near = 1;
  int far = 4;
  
  double noise_abs;
  
  /* calculate initial model of signal */
  for (i = near; i < arrx-far; i++) {
    for (j = 0; j < arry; j++) {
      sig[i*arry + j] = (0.333 * data[(i-1)*arry + j]) + 
      (0.334 * data[i*arry + j]) + 
      (0.333 * data[(i+1)*arry + j]);
    }
  }
  
  /* calculate model of readout noise */
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      noise[i*arry + j] = data[i*arry + j] - sig[i*arry + j];
      noise_abs = fabs(noise[i*arry + j]);
      
      if (noise_abs > noiseHi) {
        noise[i*arry + j] = 0;
      } else if (noise_abs > noiseLo) {
        noise[i*arry + j] = (noiseHi - noise_abs) * (noise_abs / noise[i*arry + j]);
      }
      
      /* calculate final signal value */
      sig[i*arry + j] = data[i*arry + j] - noise[i*arry + j];
    }
  }
  
  return status;
}
