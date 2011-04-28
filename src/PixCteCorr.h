/* constants describing the CTE parameters reference file */
#define NUM_PHI 9  /* number of phi values in cte params file */
#define NUM_PSI 13  /* number of psi nodes in cte params file (also # of rows in table) */
#define NUM_LOGQ 4  /* number of log q columns in psi array */

/* constants describing the CTE characterization */
#define MAX_TAIL_LEN 60  /* CTE trails are characterized out to 60 pixels */
#define MAX_PHI 99999     /* max number of phi nodes */
#define CTE_REF_ROW 2048  /* row from which CTE is measured */

/* constants for different instrument names */
#define ACSWFC 1

/* parameters of readout noise decomposition routines */
#define NOISE_MODEL 1

/* full frame size of an ACS WFC single amp image without bias regions */
#define MAX_WFC_SIZE 2048

/* temporary so things compile for testing */
#define NUM_SIM_ITER 5

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
                   double chg_leak_interp[MAX_TAIL_LEN*NUM_LOGQ],
                   double chg_open_interp[MAX_TAIL_LEN*NUM_LOGQ]);
int InterpolatePhi(const double dtde_l[NUM_PHI], const int q_dtde[NUM_PHI],
                   const int shft_nit, double dtde_q[MAX_PHI]);
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
