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
                const int ycte_qmax, const int q_pix_array[MAX_PHI], 
                const double chg_leak[MAX_TAIL_LEN*ycte_qmax],
                const double chg_open[MAX_TAIL_LEN*ycte_qmax]);
double eval_prof(const int ycte_qmax, const double chg[MAX_TAIL_LEN*ycte_qmax],
                 const int t, const int q, const double val4tmax);
int eval_qpix(const int q_pix_array[MAX_PHI], const double p);


int FixYCte(const int arrx, const int arry, const double sig_cte[arrx*arry],
            double sig_cor[arrx*arry], const int ycte_qmax, const int q_pix[MAX_PHI],
            const double chg_leak[MAX_TAIL_LEN*ycte_qmax], 
            const double chg_open[MAX_TAIL_LEN*ycte_qmax],
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
    
    for (n = 0; n < NUM_SIM_ITER; n++) {
      status = sim_readout(arrx, pix_cur, pix_read, ycte_qmax, q_pix, chg_leak, chg_open);
      
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

/* simulate readout of a column */
int sim_readout(const int arrx, double pix_cur[arrx], double pix_read[arrx],
                const int ycte_qmax, const int q_pix_array[MAX_PHI], 
                const double chg_leak[MAX_TAIL_LEN*ycte_qmax],
                const double chg_open[MAX_TAIL_LEN*ycte_qmax]) {
  
  /* status variable for return */
  int status = 0;
  
  /* iteration variables */
  int i, t, q, qpix;
  
  int trapq[ycte_qmax];  
  int trapq_init = 100;
  
  double iy, free1, fill;
  
  /* initialize trapq array */
  for (q = 0; q < ycte_qmax; q++) {
    trapq[q] = trapq_init;
  }
  
  /* go through readout cycle. read the end pixel first and put it next
   * onto the buffer. then shift the others over. do this arrx times .*/
  for (i = 0; i < arrx; i++) {
    
    iy = (i + 1.0) / MAX_WFC_SIZE;
    
    /* first half of the loop deals with the emptying of traps.
     * trapq[q] tells you how long it has been since trap #q
     * has released its charge. */
    free1 = 0.0;
    q = 0;
    while (q < ycte_qmax && trapq[q] < trapq_init) {
      
      t = trapq[q] + 1;
      if (t <= trapq_init) {
        /* once a trap has given off its electrons for this shift, then
         * increment how long it's been since it was filled. */
        
        free1 += eval_prof(ycte_qmax, chg_leak, t-1, q, 0.0) * iy;
        
        trapq[q]++;
      }
      
      q++;
    } /* end loop emptying traps */
    
    pix_read[i] = pix_cur[i] + free1;
    
    /* second half of the loop deals with the filling of traps.
     * it is easier to confine fewer electrons. a larger electron
     * cloud sees more traps as a cloud but less traps per electron.
     * trap density is probably uniform but electron cloud is not.*/
    fill = 0.0;
    qpix = eval_qpix(q_pix_array, pix_read[i]);
    for (q = 0; q < qpix; q++) {
      t = trapq[q];
      fill += eval_prof(ycte_qmax, chg_open, t-1, q, 1.0) * iy;
      trapq[q] = 0;
    } /* end loop filling traps */
    
    pix_read[i] -= fill;
  } /* end loop over columns */
  
  return status;
}

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
