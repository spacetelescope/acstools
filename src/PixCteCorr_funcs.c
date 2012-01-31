#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "PixCteCorr.h"

#define ERROR_RETURN 2

/*
 * CalcCteFrac calculates the multiplicative factor that accounts for the
 * worsening of CTE over time.
 */
double CalcCteFrac(const double expstart, const double scalemjd[NUM_SCALE],
                   const double scaleval[NUM_SCALE]) {
  
  /* iteration variables */
  int i;
  
  /* variables used to calculate the CTE scaling slope */
  double mjd_pt1 = 0;
  double mjd_pt2 = 0;      /* the MJD points at which the scaling is defined */
  double cte_pt1, cte_pt2; /* the CTE frac points at which the scaling is defined */
  
  /* return value */
  double cte_frac;
  
  /* find the values that bound this exposure */
  for (i = 0; i < NUM_SCALE-1; i++) {
    if (expstart >= scalemjd[i] && expstart < scalemjd[i+1]) {
      mjd_pt1 = scalemjd[i];
      mjd_pt2 = scalemjd[i+1];
      cte_pt1 = scaleval[i];
      cte_pt2 = scaleval[i+1];
      break;
    }
  }
  
  /* it's possible this exposure is not bounded by any of defining points,
   * in that case we're extrapolating based on the last two points. */
  if (expstart >= scalemjd[NUM_SCALE-1] && mjd_pt1 == 0 && mjd_pt2 == 0) {
    mjd_pt1 = scalemjd[NUM_SCALE-2];
    mjd_pt2 = scalemjd[NUM_SCALE-1];
    cte_pt1 = scaleval[NUM_SCALE-2];
    cte_pt2 = scaleval[NUM_SCALE-1];
  } else if (mjd_pt1 == 0 && mjd_pt2 == 0) {
    return (cte_frac = -9999.0);
  }
  
  cte_frac = ((cte_pt2 - cte_pt1) / (mjd_pt2 - mjd_pt1)) * (expstart - mjd_pt1) + cte_pt1;
  
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
 * Input dtde_l is read from the CTE parameters file.
 *
 * Outputs dtde_q is arrays MAX_PHI long (should be 99999).
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
  
  for (p = 0; p < NUM_PHI-1; p++) {
    log_qa = log10((double) q_dtde[p]);
    log_qb = log10((double) q_dtde[p+1]);
    log_da = log10(dtde_l[p]);
    log_db = log10(dtde_l[p+1]);
    
    for (q = q_dtde[p]; q < q_dtde[p+1]; q++) {
      interp_pt = log10((double) q);
      interp_dist = (interp_pt - log_qa) / (log_qb - log_qa);
      interp_val = log_da + (interp_dist * (log_db - log_da));
      
      dtde_q[q-1] = pow(10, interp_val);
      if (p == NUM_PHI - 2) {
        dtde_q[q-1] = dtde_l[p] * (1.0 - (double) (q - q_dtde[p]) / (double) (q_dtde[p+1] - q_dtde[p]));
      }
    }
  }
  
  dtde_q[MAX_PHI - 1] = 0.0;
  
  for (q = 0; q < MAX_PHI; q++) {
    dtde_q[q] = 1.0 - pow(1.0 - dtde_q[q]/(double) CTE_REF_ROW, 
                          (double) CTE_REF_ROW/(double) shft_nit);
  }
  
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
                    double dpde_l[NUM_LEV]) {
  
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
  
  return status;
}


/*
 * Attempt to separate readout noise from signal, since CTI happens before
 * readout noise is added to the signal.
 *
 * The clipping parameter read_noise controls the maximum amount by which a pixel
 * will be modified, or the maximum amplitude of the read noise.
 */
int DecomposeRN(const int arrx, const int arry, const double data[arrx*arry],
                const double read_noise, const int noise_model,
                double sig_arr[arrx*arry], double noise_arr[arrx*arry]) {
  
  /* status variable for return */
  int status = 0;
  
  /* iteration variables */
  int i, j, i2, j2, it_count;
  
  int max_it1 = 25;
  int max_it2 = 30;
  
  /* accumulation variables */
  double sum;
  int num;
  
  double d1, d2, d3;
  
  double rms;
  
  double * local_sigma;
  double * local_weight;
  char * local_mask;
  
  /* check for valid noise_model */
  if (noise_model != 0 && noise_model != 1 && noise_model != 2) {
    return (status = ERROR_RETURN);
  }
  
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      sig_arr[i*arry + j] = data[i*arry + j];
      noise_arr[i*arry + j] = 0.0;
    }
  }
  
  /* no smoothing */
  if (noise_model == 0) {
    return status;
  }
  
  /* adjust each pixel to be more similar to the three pixels below it */
  it_count = 0;
  
  do {
    sum = 0.0;
    num = 0;
    
    for (i = 3; i < arrx; i++) {
      for (j = 0; j < arry; j++) {
        d1 = sig_arr[i*arry + j] - sig_arr[(i-1)*arry + j];
        
        if (d1 > 0.1*read_noise) {
          d1 = 0.1*read_noise;
        } else if (d1 < -0.1*read_noise) {
          d1 = -0.1*read_noise;
        }
        
        d2 = sig_arr[i*arry + j] - sig_arr[(i-2)*arry + j];
        
        if (d2 > 0.1*read_noise) {
          d2 = 0.1*read_noise;
        } else if (d2 < -0.1*read_noise) {
          d2 = -0.1*read_noise;
        }
        
        d3 = sig_arr[i*arry + j] - sig_arr[(i-3)*arry + j];
        
        if (d3 > 0.1*read_noise) {
          d3 = 0.1*read_noise;
        } else if (d3 < -0.1*read_noise) {
          d3 = -0.1*read_noise;
        }
        
        noise_arr[i*arry + j] += (d1 + d2 + d3) / 3.0;
        
        sum += pow(noise_arr[i*arry + j], 2);
        num++;
      }
    }
    
    for (i = 0; i < arrx; i++) {
      for (j = 0; j < arry; j++) {
        sig_arr[i*arry + j] = data[i*arry + j] - noise_arr[i*arry + j];
      }
    }
    
    rms = sqrt(sum / (double) num);
    
    it_count++;
    
  } while (it_count < 25 || (it_count < 30 && rms < 1.25*read_noise));
  
  /* allocate sigma, weight, and mask arrays */
  local_sigma = (double *) malloc(arrx * arry * sizeof(double));
  local_weight = (double *) malloc(arrx * arry * sizeof(double));
  local_mask = (char *) malloc(arrx * arry * sizeof(char));
  
  /* calculate local sigma, initialize weights and mask */
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      sum = 0.0;
      
      /* add up local pixels to calculate sigma */
      for (i2 = fmaxl(0, i-1); i2 <= fminl(i+1, arrx-1); i2++) {
        for (j2 = fmaxl(0, j-1); j2 <= fminl(j+1, arry-1); j2++) {
          sum += pow(noise_arr[i2*arry + j2], 2);
        }
      }
      
      local_sigma[i*arry + j] = sqrt(sum / 9.0);
      
      if (local_sigma[i*arry + j] > 1.72*read_noise) {
        local_weight[i*arry + j] = 1.72 * read_noise / local_sigma[i*arry + j];
        
        local_mask[i*arry + j] = 0;
        
      } else {
        local_weight[i*arry + j] = 1.0;
        
        local_mask[i*arry + j] = 1;
      }
    }
  }
  
  if (noise_model == 1) {
    /* downweight pixels with locally high noise */
    for (i = 1; i < arrx-1; i++) {
      for (j = 1; j < arry-1; j++) {
        num = 0;
        
        for (i2 = i-1; i2 <= i+1; i2++) {
          for (j2 = j-1; j2 <= j+1; j2++) {
            num += (int) local_mask[i2*arry + j2];
          }
        }
        
        noise_arr[i*arry + j] *= (double) num / 9.0;
      }
    }
    
    for (i = 0; i < arrx; i++) {
      for (j = 0; j < arry; j++) {
        sig_arr[i*arry + j] = data[i*arry + j] - noise_arr[i*arry + j];
      }
    }
    
  } else if (noise_model == 2) {
    /* downweight pixels with locally high noise, but less so than under
     * noise_model == 1.
     */
    for (i = 1; i < arrx-1; i++) {
      for (j = 1; j < arry-1; j++) {
        sum = 0.0;
        
        for (i2 = i-1; i2 <= i+1; i2++) {
          for (j2 = j-1; j2 <= j+1; j2++) {
            sum += local_weight[i2*arry + j2];
          }
        }
        
        noise_arr[i*arry + j] *= sum / 9.0;
      }
    }
    
    for (i = 0; i < arrx; i++) {
      for (j = 0; j < arry; j++) {
        sig_arr[i*arry + j] = data[i*arry + j] - noise_arr[i*arry + j];
      }
    }
  }
  
  free(local_sigma);
  free(local_weight);
  free(local_mask);
  
  return status;
}
