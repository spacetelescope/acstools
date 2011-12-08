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
                const double read_noise, double sig_arr[arrx*arry], 
                double noise_arr[arrx*arry]) {
  
  /* status variable for return */
  int status = 0;
  
  /* iteration variables */
  int i, j, i2, j2, k;
  
  /* maximum number of smoothing iterations */
  int max_nits = 20;
  
  /* pixel mask. mask = 0 means don't modify this pixel */
  char * mask;
  
  /* number of pixels modified in a smoothing iteration */
  int num_mod;
  
  /* generic containers for adding things up */
  double sum;
  int num;
  
  double mean;
  double diff;
  
  /* function prototypes */
  int mask_stars(const int arrx, const int arry, const double data[arrx*arry],
                 char mask[arrx*arry], const double read_noise);
  
  /* allocate pixel mask */
  mask = (char *) malloc(arrx * arry * sizeof(char));
  
  /* flag high signal pixels that we shouldn't modify with 0 */
  mask_stars(arrx, arry, data, mask, read_noise);
  
  /* initialize signal and noise arrays */
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      sig_arr[i*arry + j] = data[i*arry + j];
      noise_arr[i*arry + j] = 0;
    }
  }
  
  /* if the read_noise is 0 we shouldn't do anything, just return with
   * signal = data, noise = 0. */
  if (read_noise == 0) {
    return status;
  }
  
  /* now perform the actual smoothing adjustments */
  for (k = 0; k < max_nits; k++) {
    num_mod = 0;
    
    for (i = 2; i < arrx-2; i++) {
      for (j = 2; j < arry-2; j++) {
        if (mask[i*arry + j] == 0) {
          continue;
        }
        
        sum = 0.0;
        num = 0;
        
        /* calculate local mean of non-maxima pixels */
        for (i2 = i-1; i2 <= i+1; i2++) {
          for (j2 = j-1; j2 <= j+1; j2++) {
            if (mask[i2*arry + j2] == 1) {
              sum += sig_arr[i2*arry + j2];
              num++;
            }
          }
        }
        
        /* if we have enough local non-maxima pixels calculate a readnoise
         * correction that brings this pixel closer to the mean */
        if (num >= 4) {
          mean = sum / (double) num;
          
          diff = sig_arr[i*arry + j] - mean;
          
          /* clip the diff so we don't modify this pixel too much */
          if (diff > 0.1*read_noise) {
            diff = 0.1*read_noise;
          } else if (diff < -0.1*read_noise) {
            diff = -0.1*read_noise;
          }
          
          if (diff != 0) {
            num_mod++;
          }
          
          noise_arr[i*arry + j] += diff * pow((double) num/9.0, 2);
        }
      }
    }
    
    /* calculate the smoothing correction and the rms of the read noise image */
    sum = 0.0;
    
    for (i = 2; i < arrx-2; i++) {
      for (j = 2; j < arry-2; j++) {
        sig_arr[i*arry + j] = data[i*arry + j] - noise_arr[i*arry + j];
        
        sum += pow(noise_arr[i*arry + j], 2);
      }
    }
    
    /* if the rms of the noise is greater than the max read noise, we're done */
    if (sqrt(sum / (double) num_mod) > read_noise) {
      break;
    }
  }
  
  free(mask);
  
  return status;
}


/*
 * Mask high signal pixels so they aren't clipped by the readnoise
 * decomposition.
 */
int mask_stars(const int arrx, const int arry, const double data[arrx*arry],
               char mask[arrx*arry], const double read_noise) {
  
  int status = 0;
  
  /* iteration variables */
  int i, j, i2, j2, i2_start, i3, j3;
  
  /* need a second copy of the mask array so one can stay static while
   * the other is modified */
  char * mask_copy;
  
  /* a mask of warm columns */
  char * warm_mask;
  
  int high_count;
  
  /* flag for breaking out of loops */
  short int break_out;
  
  /* holder values for mean of pixels surrounding a maximum and the
   * difference between the pixel and the mean */
  double surr_mean;
  double smean_diff1, smean_diff2, smean_diff2_temp, smean_diff3;
  
  double dist1, dist2;
  
  /* allocate arrays */
  mask_copy = (char *) malloc(arrx * arry * sizeof(char));
  warm_mask = (char *) malloc(arrx * arry * sizeof(char));
  
  /* initialize arrays */
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      mask_copy[i*arry + j] = 1;
      warm_mask[i*arry + j] = 0;
    }
  }
  
  /* set edges to no modification */
  for (i = 0; i < arrx; i++) {
    mask_copy[i*arry + 0] = 0;
    mask_copy[i*arry + 1] = 0;
    mask_copy[i*arry + (arry-1)] = 0;
    mask_copy[i*arry + (arry-2)] = 0;
  }
  for (j = 0; j < arry; j++) {
    mask_copy[0*arry + j] = 0;
    mask_copy[1*arry + j] = 0;
    mask_copy[(arrx-1)*arry + j] = 0;
    mask_copy[(arrx-2)*arry + j] = 0;
  }
  
  /* identify warm columns */
  for (j = 2; j < arry-2; j++) {
    for (i = 2; i < arrx-2; i++) {
      high_count = 0;
      
      i2_start = (int) fminl(i+1, arrx-101);
      
      for (i2 = i2_start; i2 <= i2_start+100; i2++) {
        if (data[i2*arry + j] > data[i2*arry + (j-1)] &&
            data[i2*arry + j] > data[i2*arry + (j+1)]) {
          high_count++;
        }
      }
      
      if (high_count > 90) {
        warm_mask[i*arry + j] = 1;
      }
    }
  }
  
  /* find local maxima */
  for (i = 2; i < arrx-2; i++) {
    for (j = 2; j < arry-2; j++) {
      /* compare this pixel to its neighbors, if it's lower than any of
       * them then move on to the next pixel */
      break_out = 0;
      
      for (i2 = i-1; i2 <= i+1; i2++) {
        for (j2 = j-1; j2 <= j+1; j2++) {
          if (data[i*arry + j] < data[i2*arry + j2]) {
            break_out = 1;
            break;
          }
        }
        
        if (break_out) {
          break;
        }
      }
      
      if (break_out) {
        continue;
      }
      
      /* find the difference between this pixel and the mean of its neighbors
       * for a one pixel aperture, then for two and three pixel apertures */
      surr_mean = data[(i-1)*arry + (j+1)] + data[(i+0)*arry + (j+1)] +
                  data[(i+1)*arry + (j+1)] + data[(i-1)*arry + (j+0)] +
                  data[(i+1)*arry + (j+0)] + data[(i-1)*arry + (j-1)] +
                  data[(i+0)*arry + (j-1)] + data[(i+1)*arry + (j-1)];
      surr_mean /= 8.0;
      
      smean_diff1 = data[i*arry + j] - surr_mean;
      
      /* two pixel aperture */
      smean_diff2 = 0.0;
      
      for (i2 = i-1; i2 <= i; i2++) {
        for (j2 = j-1; j2 <= j; j2++) {
          surr_mean = data[(i2-1)*arry + (j2-1)] + data[(i2-1)*arry + (j2+0)] +
                      data[(i2-1)*arry + (j2+1)] + data[(i2-1)*arry + (j2+2)] +
                      data[(i2+0)*arry + (j2+2)] + data[(i2+1)*arry + (j2+2)] +
                      data[(i2+2)*arry + (j2+2)] + data[(i2+2)*arry + (j2+1)] +
                      data[(i2+2)*arry + (j2+0)] + data[(i2+2)*arry + (j2-1)] +
                      data[(i2+1)*arry + (j2-1)] + data[(i2+0)*arry + (j2-1)];
          surr_mean /= 12.0;
          
          smean_diff2_temp = data[(i2+0)*arry + (j2+0)] + data[(i2+1)*arry + (j2+0)] +
                             data[(i2+0)*arry + (j2+1)] + data[(i2+1)*arry + (j2+1)];
          smean_diff2_temp -= (4.0 * surr_mean);
          
          smean_diff2 = fmax(smean_diff2, smean_diff2_temp);
        }
      }
      
      /* three pixle aperture */
      surr_mean = data[(i-2)*arry + (j-2)] + data[(i-2)*arry + (j-1)] +
                  data[(i-2)*arry + (j+0)] + data[(i-2)*arry + (j+1)] +
                  data[(i-2)*arry + (j+2)] + data[(i-1)*arry + (j+2)] +
                  data[(i+0)*arry + (j+2)] + data[(i+1)*arry + (j+2)] +
                  data[(i+2)*arry + (j-2)] + data[(i+2)*arry + (j-1)] +
                  data[(i+2)*arry + (j+0)] + data[(i+2)*arry + (j+1)] +
                  data[(i+2)*arry + (j+2)] + data[(i+1)*arry + (j-2)] +
                  data[(i+0)*arry + (j-2)] + data[(i-1)*arry + (j-2)];
      surr_mean /= 16.0;
      
      smean_diff3 = data[(i-1)*arry + (j+1)] + data[(i+0)*arry + (j+1)] +
                    data[(i+1)*arry + (j+1)] + data[(i-1)*arry + (j+0)] +
                    data[(i+0)*arry + (j+0)] + data[(i+1)*arry + (j+0)] +
                    data[(i-1)*arry + (j-1)] + data[(i+0)*arry + (j-1)] +
                    data[(i+1)*arry + (j-1)];
      smean_diff3 -= (9.0 * surr_mean);
      
      if (smean_diff1 > 5.0  * read_noise * (1+warm_mask[i*arry + j]) ||
          smean_diff2 > 7.5  * read_noise * (1+warm_mask[i*arry + j]) ||
          smean_diff3 > 10.0 * read_noise * (1+warm_mask[i*arry + j])) {
        mask_copy[i*arry + j] = 0;
      }
    }
  }
  
  /* copy the mask_copy to the mask */
  for (i = 0; i < arrx; i++) {
    for (j = 0; j < arry; j++) {
      mask[i*arry + j] = mask_copy[i*arry + j];
    }
  }
  
  /* having found maxima, identify pixels associated with those maxima */
  for (i = 2; i < arrx-2; i++) {
    for (j = 2; j < arry-2; j++) {
      if (mask_copy[i*arry + j] == 1) {
        continue;
      }
      
      high_count = 0;
      
      for (i2 = i-5; i2 <= i+5; i2++) {
        for (j2 = j-5; j2 <= j+5; j2++) {
          /* don't go outside the array */
          if (i2 < 0 || i2 >= arrx) {
            break;
          } else if (j2 < 0 || j2 >= arry) {
            continue;
          }
          
          dist1 = sqrt(pow(i2-i,2) + pow(j2-j,2));
          
          if (mask[i2*arry + j2] == 1 && dist1 <= 5.5) {
            for (i3 = i2-1; i3 <= i2+1; i3++) {
              for (j3 = j2-1; j3 <= j2+1; j3++) {
                break_out = 0;
                
                /* don't go outside the array */
                if (i3 < 0 || i3 >= arrx || j3 < 0 || j3 >= arry) {
                  break_out = 1;
                  break;
                }
                
                dist2 = sqrt(pow(i3-i,2) + pow(j3-j,2));
                
                if (dist2 < dist1 && mask[i3*arry + j3] == 1) {
                  break_out = 1;
                  break;
                } else if (dist2 < dist1-0.5 && 
                           data[i3*arry + j3] < data[i2*arry + j2]) {
                  break_out = 1;
                  break;
                } else if (dist2 > dist1+0.5 &&
                           data[i3*arry + j3] > data[i2*arry + j2]) {
                  break_out = 1;
                  break;
                }
              }
              
              if (break_out) {
                break;
              }
            }
            
            if (break_out) {
              continue;
            }
            
            mask[i2*arry + j2] = 0;
            high_count++;
          }
        }
      }
      
      /* if more than 1 pixel has had its mask modified repeat the loop
       * for this pixel */
      if (high_count > 1) {
        j--;
      }
    }
  }
  
  /* now make sure warm columns are masked out */
  for (i = 2; i < arrx-2; i++) {
    for (j = 2; j < arry-2; j++) {
      if (warm_mask[i*arry + j] == 1) {
        mask[i*arry + j] = 0;
      }
    }
  }
  
  free(mask_copy);
  free(warm_mask);
  
  return status;
}
