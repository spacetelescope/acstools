#include <stdio.h>
#include <stdlib.h>
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
  double pix_obs[arrx];
  double pix_cur[arrx];
  double pix_read[arrx];
  
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
      pix_obs[j] = sig_cte[j*arry + i];
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
      sig_cor[j*arry + i] = pix_cur[j];
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

int AddYCte(const int arrx, const int arry, const double sig_cte[arrx*arry],
            double sig_cor[arrx*arry], const double cte_frac,
            const int shft_nit, const int levels[NUM_LEV],
            const double dpde_l[NUM_LEV], const int tail_len[NUM_LEV],
            const double chg_leak_lt[MAX_TAIL_LEN*NUM_LEV],
            const double chg_open_lt[MAX_TAIL_LEN*NUM_LEV]) {
  
  /* status variable for return */
  int status = 0;
  
  /* iteration variables */
  int i, j, n;
  
  /* arrays to hold columns of data */
  double pix_obs[arrx];
  double pix_cur[arrx];
  double pix_read[arrx];
  
  /* loop over columns. columns are independent of each other. */
  for (i = 0; i < arry; i++) {    
    /* copy column data */
    for (j = 0; j < arrx; j++) {
      pix_obs[j] = sig_cte[j*arry + i];
      pix_cur[j] = pix_obs[j];
      pix_read[j] = pix_obs[j];
    }
    
    status = sim_readout_nit(arrx, pix_cur, pix_read, cte_frac, shft_nit,
                             levels, dpde_l, tail_len, chg_leak_lt, chg_open_lt);
    if (status != 0) {
      return status;
    }
    
    /* copy blurred column to output */
    for (j = 0; j < arrx; j++) {
      sig_cor[j*arry + i] = pix_read[j];
    }
    
  } /* end loop over columns */
  
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
  double * ftrap_lj;
  
  double pix0;    /* current pixel container */
  double fpix;    /* fraction of this pixel involved */
  double fopn;    /* fraction of this trap that is open */
  double ffil;    /* fraction of this trap that gets filled */
  double dpix;    /* amount of charge that gets transferred */
  
  /* calloc initializes array members to zero */
  ftrap_lj = calloc(arrx*NUM_LEV, sizeof(double));
  
  /* copy input to output */
  for (i = 0; i < arrx; i++) {
    pix_read[i] = pix_cur[i];
  }
  
  /* iterate over every pixel in the column */
  for (i = 0; i < arrx; i++) {
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
      dpix = cte_frac * dpde_l[l] * ffil * ((double) (i+1) / (double) CTE_REF_ROW);
      
      /* remove electrons from the pixel */
      pix_read[i] -= dpix;
      
      /* redistribute electrons in the tail */
      if ((i + tail_len[l]) < arrx) {
        tmax = tail_len[l];
      } else {
        tmax = arrx - i - 1;
      }

      for (t = 1; t <= tmax; t++) {
        pix_read[i+t] += (dpix * chg_leak_lt[(t-1)*NUM_LEV + l]);
        ftrap_lj[(i+t)*NUM_LEV + l] += (ffil * chg_open_lt[(t-1)*NUM_LEV + l]);
      }
    }
  }
  
  free(ftrap_lj);
  
  return status;
}
