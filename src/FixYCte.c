#include <stdio.h>
#include <math.h>
#include <string.h>

#include "PixCteCorr.h"

/* constants that come up for optional logfile */
#define ENDLINE "\n"
#define J_PRINT_MIN 1996 // 1997-1
#define J_PRINT_MAX 2015 // 2015-0

/* function prototypes */
int sim_readout(const int arrx, double pix_cur[arrx], double pix_read[arrx],
                const double cte_frac, const int levels[NUM_LEV],
                const double dpde_l[NUM_LEV], const int tail_len[NUM_LEV],
                const double chg_leak_lt[MAX_TAIL_LEN*NUM_LEV],
                const double chg_open_lt[MAX_TAIL_LEN*NUM_LEV]);
int sim_readout_nit(const int arrx, double pix_cur[arrx], double pix_read[arrx],
                    const double cte_frac, const int shft_nit, const int levels[NUM_LEV],
                    const double dpde_l[NUM_LEV], const int tail_len[NUM_LEV],
                    const double chg_leak_lt[MAX_TAIL_LEN*NUM_LEV],
                    const double chg_open_lt[MAX_TAIL_LEN*NUM_LEV]);
//int sim_readout(const int arrx, double pix_cur[arrx], double pix_read[arrx],
//                const int ycte_qmax, const int q_pix_array[MAX_PHI], 
//                const double chg_leak[MAX_TAIL_LEN*ycte_qmax],
//                const double chg_open[MAX_TAIL_LEN*ycte_qmax]);
//double eval_prof(const int ycte_qmax, const double chg[MAX_TAIL_LEN*ycte_qmax],
//                 const int t, const int q, const double val4tmax);
//int eval_qpix(const int q_pix_array[MAX_PHI], const double p);


int FixYCte(const int arrx, const int arry, const double sig_cte[arrx*arry],
            double sig_cor[arrx*arry], const double cte_frac, const int sim_nit,
            const int shft_nit, const int levels[NUM_LEV],
            const double dpde_l[NUM_LEV], const int tail_len[NUM_LEV],
            const double chg_leak_lt[MAX_TAIL_LEN*NUM_LEV],
            const double chg_open_lt[MAX_TAIL_LEN*NUM_LEV],
            const char *amp_name, const char *log_file) {
  
  /* status variable for return */
  int status = 0;
  
  /* iteration variables */
  int i, j, n;
  
  /* arrays to hold columns of data */
  double pix_obs[arry];
  double pix_cur[arry];
  double pix_read[arry];
  
  /* optional log file */
  int doLog = 0;
  FILE *flog;
  
  /* Open optional log file */
  if (strlen(log_file) > 0) {
    doLog = 1;
    flog = fopen(log_file, "a"); /* Always append */
    
    /* Header */
    fprintf(flog, "#%s# AMP: %s%s", ENDLINE, amp_name, ENDLINE);
    fprintf(flog, "#%-4s ", "#PIX");
    for (j=J_PRINT_MIN; j<J_PRINT_MAX; j++) {
      fprintf(flog, "Y=%-4i ", j+1);
    }
    fprintf(flog, "%s", ENDLINE);
  }
  
  /* loop over columns. columns are independent of each other. */
  for (i = 0; i < arry; i++) {
    
    /* copy column data */
    for (j = 0; j < arrx; j++) {
      pix_obs[j] = sig_cte[j*arrx + i];
      pix_cur[j] = pix_obs[j];
      pix_read[j] = 0.0;
    }
    
    for (n = 0; n < sim_nit; n++) {
      status = sim_readout_nit(arrx, pix_cur, pix_read, cte_frac, shft_nit,
                               levels, dpde_l, tail_len, chg_leak_lt, chg_open_lt);
      if (status != 0) {
        return status;
      }
      
      for (j = 0; j < arrx; j++) {
        pix_cur[j] += pix_obs[j] - pix_read[j];
      }
    }
    
    /* copy fixed column to output */
    for (j = 0; j < arrx; j++) {
      sig_cor[j*arrx + i] = pix_cur[j];
    }
    
    /* Write to log file */
    if (doLog) {
	    fprintf(flog, "# X=%-i%s", i+1, ENDLINE);
      
      // PIX_CURR
	    fprintf(flog, "%-4s ", "CURR");
      for (j=J_PRINT_MIN; j<J_PRINT_MAX; j++) {
        fprintf(flog, "%6i ", (int)(pix_cur[j]+0.5));
      }
	    fprintf(flog, "%s", ENDLINE);
      
	    // PIX_OBSD
	    fprintf(flog, "%-4s ", "OBSD");
      for (j=J_PRINT_MIN; j<J_PRINT_MAX; j++) {
        fprintf(flog, "%6i ", (int)(pix_obs[j]+0.5));
      }
	    fprintf(flog, "%s", ENDLINE);
      
	    // PIX_READ
	    fprintf(flog, "%-4s ", "READ");
      for (j=J_PRINT_MIN; j<J_PRINT_MAX; j++) {
        fprintf(flog, "%6i ", (int)(pix_read[j]+0.5));
      }
	    fprintf(flog, "%s", ENDLINE);
    }
  } /* end loop over columns */
  
  /* Close optional log file */
  if (doLog) {
    fclose(flog);
  }
  
  return status;
}

/* call sim_readout shft_nit times as per Jay's new algorithm */
int sim_readout_nit(const int arrx, double pix_cur[arrx], double pix_read[arrx],
                    const double cte_frac, const int shft_nit, const int levels[NUM_LEV],
                    const double dpde_l[NUM_LEV], const int tail_len[NUM_LEV],
                    const double chg_leak_lt[MAX_TAIL_LEN*NUM_LEV],
                    const double chg_open_lt[MAX_TAIL_LEN*NUM_LEV]) {
  /* status variable for return */
  int status = 0;
  
  /* iteration variables */
  int i,j;
  
  /* local container of column data */
  double pix_local[arrx];
  
  for (i = 0; i < arrx; i++) {
    pix_local[i] = pix_cur[i];
  }
  
  for (j = 0; j < shft_nit; j++) {
    status = sim_readout(arrx, pix_local, pix_read, cte_frac, levels, dpde_l,
                         tail_len, chg_leak_lt, chg_open_lt);
    if (status != 0) {
      return status;
    }
    
    /* don't need to copy this back the last time through */
    if (j < shft_nit - 1) {
      for (i = 0; i < arrx; i++) {
        pix_local[i] = pix_read[i];
      }
    }
  }
  
  return status;
}
  

/* workhorse function that moves simulates CTE by shifting charge down the column
 * and keeping track of charge trapped and released */
int sim_readout(const int arrx, double pix_cur[arrx], double pix_read[arrx],
                const double cte_frac, const int levels[NUM_LEV],
                const double dpde_l[NUM_LEV], const int tail_len[NUM_LEV],
                const double chg_leak_lt[MAX_TAIL_LEN*NUM_LEV],
                const double chg_open_lt[MAX_TAIL_LEN*NUM_LEV]) {
  
  /* status variable for return */
  int status = 0;
  
  /* iteration variables */
  int i,l,t;
  int tmax;
  
  /* holds some trap info, I guess */
  double ftrap_lj[arrx*NUM_LEV] = {0.0};  /* all elemnts initialized to 0 */
  
  double pix0;    /* current pixel container */
  double fpix;    /* fraction of this pixel involved */
  double fopn;    /* fraction of this trap that is open */
  double ffil;    /* fraction of this trap that gets filled */
  double dpix;    /* amount of charge that gets transferred */
  
  /* copy input to output */
  for (i = 0; i < arrx; i++) {
    pix_read[i] = pix_cur[i];
  }
  
  /* iterate over every pixel in the column */
  for (i = 0; i < arrx, i++) {
    pix0 = pix_read[i];
    
    for (l = 1; l < NUM_LEV; l++) {
      /* skip the rest of the levels if we don't have enough charge to reach
       * any more traps */
      if (pix0 < levels[l-1]) {
        continue;
      }
      
      /* can usually fill an entire trap, but if not calculate the fraction */
      fpix = 1.000;
      if (pix0 < levels[l]) {
        fpix = (pix0 - levels[l-1]) / (levels[l] - levels[l-1]);
      }
      
      /* how much of trap is available for filling */
      fopn = 1.0 - ftrap_lj[i*NUM_LEV + l];
      
      /* what fraction of the trap can I fill given current conditions? */
      ffil = fpix*fopn;
      
      /* how many electrons can this take? */
      dpix = cte_frac * dpde_l[l] * ffil * ((double) i / (double) CTE_REF_ROW);
      
      /* remove electrons from the pixel */
      pix_read[i] = pix_read[i] - dpix;
      
      /* redistribute pixels in the tail */
      if (tail_len[l] < arrx - i) {
        tmax = tail_len[l];
      } else {
        tmax = arrx - i;
      }

      for (t = 0; t < tmax; t++) {
        pix_read[i+t] = pix_read[i+t] + (dpix * chg_leak_lt[t*NUM_LEV + l]);
        ftrap_lj[i*NUM_LEV + l] = ftrap_lj[(i+t)*NUM_LEV + l] + 
                                            (ffil * chg_open_lt[t*NUM_LEV + l]);
      }
    }
  }
  
  return status;
}

/* simulate readout of a column */
//int sim_readout(const int arrx, double pix_cur[arrx], double pix_read[arrx],
//                const int ycte_qmax, const int q_pix_array[MAX_PHI], 
//                const double chg_leak[MAX_TAIL_LEN*ycte_qmax],
//                const double chg_open[MAX_TAIL_LEN*ycte_qmax]) {
//  
//  /* status variable for return */
//  int status = 0;
//  
//  /* iteration variables */
//  int i, t, q, qpix;
//  
//  int trapq[ycte_qmax];  
//  int trapq_init = 100;
//  
//  double iy, free1, fill;
//  
//  /* initialize trapq array */
//  for (q = 0; q < ycte_qmax; q++) {
//    trapq[q] = trapq_init;
//  }
//  
//  /* go through readout cycle. read the end pixel first and put it next
//   * onto the buffer. then shift the others over. do this arrx times .*/
//  for (i = 0; i < arrx; i++) {
//    
//    iy = (i + 1.0) / MAX_WFC_SIZE;
//    
//    /* first half of the loop deals with the emptying of traps.
//     * trapq[q] tells you how long it has been since trap #q
//     * has released its charge. */
//    free1 = 0.0;
//    q = 0;
//    while (q < ycte_qmax && trapq[q] < trapq_init) {
//      
//      t = trapq[q] + 1;
//      if (t <= trapq_init) {
//        /* once a trap has given off its electrons for this shift, then
//         * increment how long it's been since it was filled. */
//        
//        free1 += eval_prof(ycte_qmax, chg_leak, t-1, q, 0.0) * iy;
//        
//        trapq[q]++;
//      }
//      
//      q++;
//    } /* end loop emptying traps */
//    
//    pix_read[i] = pix_cur[i] + free1;
//    
//    /* second half of the loop deals with the filling of traps.
//     * it is easier to confine fewer electrons. a larger electron
//     * cloud sees more traps as a cloud but less traps per electron.
//     * trap density is probably uniform but electron cloud is not.*/
//    fill = 0.0;
//    qpix = eval_qpix(q_pix_array, pix_read[i]);
//    for (q = 0; q < qpix; q++) {
//      t = trapq[q];
//      fill += eval_prof(ycte_qmax, chg_open, t-1, q, 1.0) * iy;
//      trapq[q] = 0;
//    } /* end loop filling traps */
//    
//    pix_read[i] -= fill;
//  } /* end loop over columns */
//  
//  return status;
//}

double eval_prof(const int ycte_qmax, const double chg[MAX_TAIL_LEN*ycte_qmax],
                 const int t, const int q, const double val4tmax) {
  
  double rval = 0.0;
  
  int qu = q;
  int qu_max = ycte_qmax - 1;
  int t_max = MAX_TAIL_LEN - 1;
  
  if (t < 0) {
    return rval;
  } else if (t > t_max) {
    return val4tmax;
  } else if (q < 0) {
    return rval;
  }
  
  if (q > qu_max) {
    qu = qu_max;
  }
  
  rval = chg[t*ycte_qmax + qu];
  
  return rval;
}

int eval_qpix(const int q_pix_array[MAX_PHI], const double p) {
  
  int rval = 0;
  
  int ip;
  int ip_max = MAX_PHI - 1;
  
  if (p < 1) {
    return rval;
  }
  
  ip = ((int) p) - 1;
  
  if (ip > ip_max) {
    ip = ip_max;
  }
  
  rval = q_pix_array[ip];
  
  return rval;
}
