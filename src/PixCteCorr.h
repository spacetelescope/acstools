/* constants describing the CTE parameters reference file */
#define NUM_PHI 11  /* number of phi values in cte params file */
#define NUM_PSI 17  /* number of psi nodes in cte params file (also # of rows in table) */
#define NUM_LOGQ 4  /* number of log q columns in psi array */

/* constants describing the CTE characterization */
#define MAX_TAIL_LEN 100  /* CTE trails are characterized out to 100 pixels */
#define MAX_PHI 49999     /* max number of phi nodes */

/* constants for different instrument names */
#define ACSWFC 1

/* number of extensions in the CTE parameters file */
#define PCTE_NUM_EXT 1

/* parameters of readout noise decomposition routines */
#define NOISE_MODEL 1

/* number of iterations of sim_readout in FixYCte */
/* this may change to a CTE parameters file in the future */
#define NUM_SIM_ITER 4

/* full frame size of an ACS WFC single amp image */
#define MAX_WFC_SIZE 2048

/* structure to hold CTE parameters from reference file */
typedef struct {
  int rn2_nit;
  double dtde_l[NUM_PHI];
  int psi_node[NUM_PSI];
  double chg_leak[NUM_PSI * NUM_LOGQ];
} CTEParams;

/* function prototypes */
int PixCteParams (const char *filename, const double mjd, CTEParams * pars);
double CalcCteFrac(const double mjd, const int instrument);
int InterpolatePsi(const double chg_leak[NUM_PSI*NUM_LOGQ], const int psi_node[],
                   double chg_leak_interp[MAX_TAIL_LEN*NUM_LOGQ]);
int InterpolatePhi(const double dtde_l[NUM_PHI], const double cte_frac,
                   double dtde_q[MAX_PHI], int q_pix_array[MAX_PHI],
                   int pix_q_array[MAX_PHI], int *ycte_qmax);
int TrackChargeTrap(const int pix_q_array[MAX_PHI], 
                    const double chg_leak_interp[MAX_TAIL_LEN*NUM_LOGQ],
                    const int ycte_qmax, 
                    double chg_leak_tq[MAX_TAIL_LEN*ycte_qmax],
                    double chg_open_tq[MAX_TAIL_LEN*ycte_qmax]);
int DecomposeRN(const int arrx, const int arry, const double data[arrx*arry], 
                const int model, const int nitr, const double readnoise,
                double sig_arr[arrx*arry], double noise_arr[arrx*arry]);
int DecomposeRNModel1(const int arrx, const int arry, 
                      const double data[arrx*arry], const int nitr,
                      double sig[arrx*arry], double noise[arrx*arry]);
int DecomposeRNModel100(const int arrx, const int arry, const double readnoise,
                        const double data[arrx*arry],
                        double sig[arrx*arry], double noise[arrx*arry]);
int FixYCte(const int arrx, const int arry, const double sig_cte[arrx*arry],
            double sig_cor[arrx*arry], const int ycte_qmax, const int q_pix[MAX_PHI],
            const double chg_leak[MAX_TAIL_LEN*ycte_qmax], 
            const double chg_open[MAX_TAIL_LEN*ycte_qmax],
            const char *amp_name, const char *log_file);
