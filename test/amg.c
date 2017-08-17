/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (IJ_matrix interface).
 * Do `driver -help' for usage info.
 * This driver started from the driver for parcsr_linear_solvers, and it
 * works by first building a parcsr matrix as before and then "copying"
 * that matrix row-by-row into the IJMatrix interface. AJC 7/99.
 *--------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_krylov.h"

#include <assert.h>
#include <time.h>

#include "HYPRE_MatvecFunctions.h"

#ifdef __cplusplus
extern "C" {
#endif

HYPRE_Int BuildParLaplacian (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParDifConv (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildParLaplacian27pt (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr );
HYPRE_Int BuildIJLaplacian27pt (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_IJMatrix *A_ptr );
HYPRE_Int BuildParVarDifConv (HYPRE_Int argc , char *argv [], HYPRE_Int arg_index , HYPRE_ParCSRMatrix *A_ptr , HYPRE_ParVector *rhs_ptr );
HYPRE_Int AddOrRestoreAIJ( HYPRE_IJMatrix  ij_A, HYPRE_Real eps, HYPRE_Int action  );

#ifdef __cplusplus
}
#endif
#define SECOND_TIME 0
 
hypre_int
main( hypre_int argc,
      char *argv[] )
{
   HYPRE_Int                 arg_index;
   HYPRE_Int                 print_usage;
   HYPRE_Int                 build_matrix_type;
   HYPRE_Int                 build_matrix_arg_index;
   HYPRE_Int                 build_rhs_type;
   HYPRE_Int                 build_rhs_arg_index;
   HYPRE_Int                 build_src_type;
   HYPRE_Int                 build_src_arg_index;
   HYPRE_Int                 solver_id;
   HYPRE_Int                 problem_id;
   HYPRE_Int                 ioutdat;
   HYPRE_Int                 poutdat;
   HYPRE_Int                 debug_flag;
   HYPRE_Int                 ierr = 0;
   HYPRE_Int                 i, j; 
   HYPRE_Int                 max_levels = 25;
   HYPRE_Int                 num_iterations;
   HYPRE_Int                 max_iter = 1000;
   HYPRE_Int                 mg_max_iter = 100;
   HYPRE_Int                 nodal = 0;
   HYPRE_Real          norm;
   HYPRE_Real          final_res_norm;
   void               *object;

   HYPRE_IJMatrix      ij_A; 
   HYPRE_IJVector      ij_b;
   HYPRE_IJVector      ij_x;

   HYPRE_ParCSRMatrix  parcsr_A;
   HYPRE_ParVector     b;
   HYPRE_ParVector     x;

   HYPRE_Solver        amg_solver;
   HYPRE_Solver        pcg_solver;
   HYPRE_Solver        pcg_precond=NULL, pcg_precond_gotten;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                *dof_func;
   HYPRE_Int		       num_functions = 1;
   HYPRE_Int		       num_paths = 1;
   HYPRE_Int		       agg_num_levels = 1;
   HYPRE_Int		       ns_coarse = 1;

   HYPRE_Int		       time_index;
   MPI_Comm            comm = hypre_MPI_COMM_WORLD;
   HYPRE_Int first_local_row, last_local_row, local_num_rows;
   HYPRE_Int first_local_col, last_local_col, local_num_cols;

   HYPRE_Real dt = dt_inf;
   HYPRE_Int time_steps = 10;

   /* parameters for BoomerAMG */
   HYPRE_Int    P_max_elmts = 8;
   HYPRE_Int    cycle_type;
   HYPRE_Int    coarsen_type = 8;
   HYPRE_Int    measure_type = 0;
   HYPRE_Int    num_sweeps = 2;  
   HYPRE_Int    relax_type = 18;   
   HYPRE_Int    add_relax_type = 18;   
   HYPRE_Int    rap2=1;
   HYPRE_Int    keepTranspose = 0;
   HYPRE_Real   tol = 1.e-8, pc_tol = 0.;
   HYPRE_Real   atol = 0.0;

   /* parameters for GMRES */
   HYPRE_Int	    k_dim;
   /* interpolation */
   HYPRE_Int      interp_type  = 6; /* default value */
   /* aggressive coarsening */
   HYPRE_Int      agg_interp_type  = 4; /* default value */

   HYPRE_Int      print_system = 0;

   HYPRE_Int rel_change = 0;

   HYPRE_Real *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   hypre_MPI_Init(&argc, &argv);

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
   hypre_GPUInit(-1);
   //nvtxDomainHandle_t domain = nvtxDomainCreateA("Domain_A");
/*
  hypre_InitMemoryDebug(myid);
*/
   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   build_matrix_type = 4;
   build_matrix_arg_index = argc;
   build_rhs_type = 2;
   build_rhs_arg_index = argc;
   build_src_type = -1;
   build_src_arg_index = argc;
   debug_flag = 0;

   solver_id = 1;
   problem_id = 1;

   ioutdat = 3;
   poutdat = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
 
   print_usage = 0;
   arg_index = 1;

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-fromfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = -1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromparcsrfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 0;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromonecsrfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-laplacian") == 0 )
      {
         arg_index++;
         build_matrix_type      = 2;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-ij27") == 0 )
      {
         arg_index++;
         build_matrix_type      = 3;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-27pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 4;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-difconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 5;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-vardifconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 6;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rhsisone") == 0 )
      {
         arg_index++;
         build_rhs_type      = 2;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsrand") == 0 )
      {
         arg_index++;
         build_rhs_type      = 3;
         build_rhs_arg_index = arg_index;
      }    
      else if ( strcmp(argv[arg_index], "-xisone") == 0 )
      {
         arg_index++;
         build_rhs_type      = 4;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhszero") == 0 )
      {
         arg_index++;
         build_rhs_type      = 5;
         build_rhs_arg_index = arg_index;
      }    
      else if ( strcmp(argv[arg_index], "-srcisone") == 0 )
      {
         arg_index++;
         build_src_type      = 2;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcrand") == 0 )
      {
         arg_index++;
         build_src_type      = 3;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-dbg") == 0 )
      {
         arg_index++;
         debug_flag = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-max_iter") == 0 )
      {
         arg_index++;
         max_iter = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mg_max_iter") == 0 )
      {
         arg_index++;
         mg_max_iter = atoi(argv[arg_index++]);
      }

      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
      }
      else
      {
         arg_index++;
      }
   }

   /* defaults for GMRES */

   k_dim = 5;

   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-problem") == 0 )
      {
         arg_index++;
         problem_id = atoi(argv[arg_index++]);
         if (problem_id == 2) 
	 {
	    build_matrix_type = 3;
	    solver_id = 3;
	 }
         
      }
      else if ( strcmp(argv[arg_index], "-k") == 0 )
      {
         arg_index++;
         k_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-atol") == 0 )
      {
         arg_index++;
         atol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-iout") == 0 )
      {
         arg_index++;
         ioutdat  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pout") == 0 )
      {
         arg_index++;
         poutdat  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-keepT") == 0 )
      {
         arg_index++;
         keepTranspose  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/
 
   if ( (print_usage) && (myid == 0) )
   {
      hypre_printf("\n");
      hypre_printf("Usage: %s [<options>]\n", argv[0]);
      hypre_printf("\n");
      hypre_printf("  -fromfile <filename>       : ");
      hypre_printf("matrix read from multiple files (IJ format)\n");
      hypre_printf("  -fromparcsrfile <filename> : ");
      hypre_printf("matrix read from multiple files (ParCSR format)\n");
      hypre_printf("  -fromonecsrfile <filename> : ");
      hypre_printf("matrix read from a single file (CSR format)\n");
      hypre_printf("\n");
      hypre_printf("  -laplacian [<options>] : build 5pt 2D laplacian problem (default) \n");
      hypre_printf("  -27pt [<opts>]         : build 27pt 3D laplacian problem\n");
      hypre_printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
      hypre_printf("    -n <nx> <ny> <nz>    : total problem size \n");
      hypre_printf("    -P <Px> <Py> <Pz>    : processor topology\n");
      hypre_printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
      hypre_printf("    -a <ax> <ay> <az>    : convection coefficients\n");
      hypre_printf("\n");
      hypre_printf("  -rhsrand               : rhs is random vector\n");
      hypre_printf("  -rhsisone              : rhs is vector with unit components (default)\n");
      hypre_printf("  -xisone                : solution of all ones\n");
      hypre_printf("  -rhszero               : rhs is zero vector\n");
      hypre_printf("\n");
      hypre_printf("                         :    -rhsfromfile, -rhsfromonefile, -rhsrand,\n");
      hypre_printf("                         :    -rhsrand, or -xisone will be ignored\n");
      hypre_printf("\n");
      hypre_printf("  -solver <ID>           : solver ID\n");
      hypre_printf("       0=AMG               1=AMG-PCG        \n");
      hypre_printf("       2=DS-PCG            3=AMG-GMRES      \n");
      hypre_printf("       4=DS-GMRES            \n");     
      hypre_printf("\n");
      hypre_printf("  -cheby_order  <val> : set order (1-4) for Chebyshev poly. smoother (default is 2)\n");
      hypre_printf("  -cheby_fraction <val> : fraction of the spectrum for Chebyshev poly. smoother (default is .3)\n");
      hypre_printf("\n"); 
      hypre_printf("  -k   <val>             : dimension Krylov space for GMRES\n");
      hypre_printf("  -tol  <val>            : set solver convergence tolerance = val\n");
      hypre_printf("  -atol  <val>           : set solver absolute convergence tolerance = val\n");
      hypre_printf("  -max_iter  <val>       : set max iterations\n");
      hypre_printf("  -mg_max_iter  <val>    : set max iterations for mg solvers\n");
      hypre_printf("\n");
      hypre_printf("  -iout <val>            : set output flag\n");
      hypre_printf("       0=no output    1=matrix stats\n"); 
      hypre_printf("       2=cycle stats  3=matrix & cycle stats\n"); 
      hypre_printf("\n");  
      hypre_printf("  -dbg <val>             : set debug flag\n");
      hypre_printf("       0=no debugging\n       1=internal timing\n       2=interpolation truncation\n       3=more detailed timing in coarsening routine\n");
      hypre_printf("\n");
      hypre_printf("  -print                 : print out the system\n");
      hypre_printf("\n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("Running with these driver parameters:\n");
      hypre_printf("  solver ID    = %d\n\n", solver_id);
   }

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("Spatial Operator");
   hypre_BeginTiming(time_index);
   if ( build_matrix_type == 2 )
   {
      BuildParLaplacian(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 3 )
   {
      void *object;
      BuildIJLaplacian27pt(argc, argv, build_matrix_arg_index, &ij_A);
      HYPRE_IJMatrixGetObject(ij_A, &object);
      parcsr_A = (HYPRE_ParCSRMatrix) object;
   }
   else if ( build_matrix_type == 4 )
   {
      BuildParLaplacian27pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 5 )
   {
      BuildParDifConv(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 6 )
   {
      BuildParVarDifConv(argc, argv, build_matrix_arg_index, &parcsr_A, &b);
      build_rhs_type      = 6;
      build_src_type      = 5;
   }

   else
   {
      hypre_printf("You have asked for an unsupported problem with\n");
      hypre_printf("build_matrix_type = %d.\n", build_matrix_type);
      return(-1);
   }

   /*-----------------------------------------------------------
    * Copy the parcsr matrix into the IJMatrix through interface calls
    *-----------------------------------------------------------*/
   ierr = HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                              &first_local_row, &last_local_row ,
                                              &first_local_col, &last_local_col );

   local_num_rows = last_local_row - first_local_row + 1;
   local_num_cols = last_local_col - first_local_col + 1;

      /*ierr += HYPRE_ParCSRMatrixGetDims( parcsr_A, &M, &N );

      ierr += HYPRE_IJMatrixCreate( comm, first_local_row, last_local_row,
                                    first_local_col, last_local_col,
                                    &ij_A );

      ierr += HYPRE_IJMatrixSetObjectType( ij_A, HYPRE_PARCSR ); 
      num_rows = local_num_rows; */

   hypre_EndTiming(time_index);
   hypre_PrintTiming("Generate Matrix", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();

   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("RHS and Initial Guess");
   hypre_BeginTiming(time_index);

   if ( build_rhs_type == 2 )
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector has unit components\n");
         hypre_printf("  Initial guess is 0\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);

      values = hypre_CTAlloc(HYPRE_Real, local_num_rows);
      for (i = 0; i < local_num_rows; i++)
         values[i] = 1.0;
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 3 )
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector has random components and unit 2-norm\n");
         hypre_printf("  Initial guess is 0\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b); 
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* For purposes of this test, HYPRE_ParVector functions are used, but
         these are not necessary.  For a clean use of the interface, the user
         "should" modify components of ij_x by using functions
         HYPRE_IJVectorSetValues or HYPRE_IJVectorAddToValues */

      HYPRE_ParVectorSetRandomValues(b, 22775);
      HYPRE_ParVectorInnerProd(b,b,&norm);
      norm = 1./sqrt(norm);
      ierr = HYPRE_ParVectorScale(norm, b);      

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 4 )
   {
      if (myid == 0)
      {
         hypre_printf("  RHS vector set for solution with unit components\n");
         hypre_printf("  Initial guess is 0\n");
      }

      /* Temporary use of solution vector */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 1.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;

      /* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b); 
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);
      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      HYPRE_ParCSRMatrixMatvec(1.,parcsr_A,x,0.,b);

      /* Initial guess */
      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);
   }

   if ( build_src_type == 2 )
   {
      if (myid == 0)
      {
         hypre_printf("  Source vector has unit components\n");
         hypre_printf("  Initial unknown vector is 0\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);

      values = hypre_CTAlloc(HYPRE_Real, local_num_rows);
      for (i = 0; i < local_num_rows; i++)
         values[i] = 1.;
      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         0 here) is usually used as the initial guess */
      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_src_type == 3 )
   {
      if (myid == 0)
      {
         hypre_printf("  Source vector has random components in range 0 - 1\n");
         hypre_printf("  Initial unknown vector is 0\n");
      }

      /* RHS */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_b);
      values = hypre_CTAlloc(HYPRE_Real, local_num_rows);

      hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
         values[i] = hypre_Rand();

      HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_b, &object );
      b = (HYPRE_ParVector) object;

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         0 here) is usually used as the initial guess */
      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      for (i = 0; i < local_num_cols; i++)
         values[i] = 0.;
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }
   else if ( build_src_type == 5 )
   {
      if (myid == 0)
      {
         hypre_printf("  Initial guess is 0 \n");
      }

      /* Initial guess */
      HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         random in 0 - 1 here) is usually used as the initial guess */
      values = hypre_CTAlloc(HYPRE_Real, local_num_cols);
      hypre_SeedRand(myid);
      for (i = 0; i < local_num_cols; i++)
         values[i] = hypre_Rand();
      HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      hypre_TFree(values);

      ierr = HYPRE_IJVectorGetObject( ij_x, &object );
      x = (HYPRE_ParVector) object;
   }

   hypre_EndTiming(time_index);
   hypre_PrintTiming("IJ Vector Setup", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   
   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/

   if (print_system)
   {
      HYPRE_IJMatrixPrint(ij_A, "IJ.out.A");
      HYPRE_IJVectorPrint(ij_b, "IJ.out.b");
      HYPRE_IJVectorPrint(ij_x, "IJ.out.x0");

   }

   /*-----------------------------------------------------------
    * Solve the system using AMG
    *-----------------------------------------------------------*/

   if (solver_id == 0)
   {
      if (myid == 0) hypre_printf("Solver:  AMG\n");
      time_index = hypre_InitializeTiming("BoomerAMG Setup");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGCreate(&amg_solver); 
      HYPRE_BoomerAMGSetCoarsenType(amg_solver, coarsen_type);
      HYPRE_BoomerAMGSetTol(amg_solver, tol);
      HYPRE_BoomerAMGSetPMaxElmts(amg_solver, P_max_elmts);
      HYPRE_BoomerAMGSetPrintLevel(amg_solver, 3);
      HYPRE_BoomerAMGSetPrintFileName(amg_solver, "driver.out.log"); 
      HYPRE_BoomerAMGSetNumSweeps(amg_solver, num_sweeps);
      if (relax_type > -1) HYPRE_BoomerAMGSetRelaxType(amg_solver, relax_type);
      HYPRE_BoomerAMGSetDebugFlag(amg_solver, debug_flag);
      HYPRE_BoomerAMGSetAggNumLevels(amg_solver, agg_num_levels);
      HYPRE_BoomerAMGSetMaxIter(amg_solver, mg_max_iter);
      HYPRE_BoomerAMGSetRAP2(amg_solver, rap2);
      HYPRE_BoomerAMGSetKeepTranspose(amg_solver, keepTranspose);

      HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      time_index = hypre_InitializeTiming("BoomerAMG Solve");
      hypre_BeginTiming(time_index);

      HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();

      HYPRE_BoomerAMGGetNumIterations(amg_solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(amg_solver, &final_res_norm);

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("BoomerAMG Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);
      HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);
#endif

      HYPRE_BoomerAMGDestroy(amg_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG 
    *-----------------------------------------------------------*/


   if (problem_id == 1 && (solver_id == 1 || solver_id == 2) )
   {
      time_index = hypre_InitializeTiming("PCG Setup");
      hypre_BeginTiming(time_index);
 
      HYPRE_ParCSRPCGCreate(hypre_MPI_COMM_WORLD, &pcg_solver);
      HYPRE_PCGSetMaxIter(pcg_solver, max_iter);
      HYPRE_PCGSetTol(pcg_solver, tol);
      HYPRE_PCGSetTwoNorm(pcg_solver, 1);
      HYPRE_PCGSetRelChange(pcg_solver, rel_change);
      HYPRE_PCGSetPrintLevel(pcg_solver, ioutdat);
      HYPRE_PCGSetAbsoluteTol(pcg_solver, atol);

      if (solver_id == 1)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) hypre_printf("Solver: AMG-PCG\n");
         HYPRE_BoomerAMGCreate(&pcg_precond); 
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         if (relax_type > -1) HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
         HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
         HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
         HYPRE_PCGSetMaxIter(pcg_solver, mg_max_iter);
         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                             (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                             pcg_precond);
      }
      else if (solver_id == 2)
      {
         
         /* use diagonal scaling as preconditioner */
         if (myid == 0) hypre_printf("Solver: DS-PCG\n");
         pcg_precond = NULL;

         HYPRE_PCGSetPrecond(pcg_solver,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                             pcg_precond);
      }
 
      HYPRE_PCGGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten !=  pcg_precond)
      {
         hypre_printf("HYPRE_ParCSRPCGGetPrecond got bad precond\n");
         return(-1);
      }
      else 
         if (myid == 0)
            hypre_printf("HYPRE_ParCSRPCGGetPrecond got good precond\n");

      HYPRE_PCGSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, 
                     (HYPRE_Vector)b, (HYPRE_Vector)x);

      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
   
      time_index = hypre_InitializeTiming("PCG Solve");
      hypre_BeginTiming(time_index);
 
      HYPRE_PCGSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, 
                     (HYPRE_Vector)b, (HYPRE_Vector)x);
 
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
 
      HYPRE_PCGGetNumIterations(pcg_solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      HYPRE_ParVectorSetRandomValues(x, 775);
      HYPRE_PCGSetup(pcg_solver, (HYPRE_Matrix)parcsr_A, 
                     (HYPRE_Vector)b, (HYPRE_Vector)x);
      HYPRE_PCGSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, 
                     (HYPRE_Vector)b, (HYPRE_Vector)x);
#endif

      HYPRE_ParCSRPCGDestroy(pcg_solver);
 
      if (solver_id == 1)
      {
         HYPRE_BoomerAMGDestroy(pcg_precond);
      }

      if (myid == 0)
      {
         hypre_printf("\n");
         hypre_printf("Iterations = %d\n", num_iterations);
         hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
      }
 
   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES 
    *-----------------------------------------------------------*/

   if (problem_id == 2 &&(solver_id == 3 || solver_id == 4) )
   {
     time_index = hypre_InitializeTiming("GMRES Solve");
     hypre_BeginTiming(time_index);
     for (j=0; j < time_steps; j++)
     {
 
       HYPRE_ParCSRGMRESCreate(hypre_MPI_COMM_WORLD, &pcg_solver);
       HYPRE_GMRESSetKDim(pcg_solver, k_dim);
       HYPRE_GMRESSetMaxIter(pcg_solver, max_iter);
       HYPRE_GMRESSetTol(pcg_solver, tol);
       HYPRE_GMRESSetAbsoluteTol(pcg_solver, atol);
       HYPRE_GMRESSetLogging(pcg_solver, 1);
       HYPRE_GMRESSetPrintLevel(pcg_solver, ioutdat);
       HYPRE_GMRESSetRelChange(pcg_solver, rel_change);
       HYPRE_ParVectorSetRandomValues(x, 775);
 
       if (solver_id == 3)
       {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) hypre_printf("Solver: AMG-GMRES\n");

         HYPRE_BoomerAMGCreate(&pcg_precond); 
         HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         if (relax_type > -1) HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
         HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
         HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
         HYPRE_GMRESSetMaxIter(pcg_solver, mg_max_iter);
         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                               (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                               pcg_precond);
       }
       else if (solver_id == 4)
       {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) hypre_printf("Solver: DS-GMRES\n");
         pcg_precond = NULL;

         HYPRE_GMRESSetPrecond(pcg_solver,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                               (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                               pcg_precond);
       }

       HYPRE_GMRESGetPrecond(pcg_solver, &pcg_precond_gotten);
       if (pcg_precond_gotten != pcg_precond)
       {
         hypre_printf("HYPRE_GMRESGetPrecond got bad precond\n");
         return(-1);
       }
       else
         if (myid == 0)
            hypre_printf("HYPRE_GMRESGetPrecond got good precond\n");
       HYPRE_GMRESSetup
         (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
       /*hypre_EndTiming(time_index);
       hypre_PrintTiming("Setup phase times", hypre_MPI_COMM_WORLD);
       hypre_FinalizeTiming(time_index);
       hypre_ClearTiming();
   
       time_index = hypre_InitializeTiming("GMRES Solve");
       hypre_BeginTiming(time_index); */
 
       HYPRE_GMRESSolve
         (pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 
       HYPRE_GMRESGetNumIterations(pcg_solver, &num_iterations);
       HYPRE_GMRESGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);
       if (myid == 0)
       {
         hypre_printf("\n");
         hypre_printf("GMRES Iterations = %d\n", num_iterations);
         hypre_printf("Final GMRES Relative Residual Norm = %e\n", final_res_norm);
         hypre_printf("\n");
       }
      
       for (i=0; i < 4; i++)
       {
         HYPRE_Int eps = 0.01;
         HYPRE_Int gmres_iter = 6-i;
         AddOrRestoreAIJ(ij_A, eps, 1);
         HYPRE_GMRESSetMaxIter(pcg_solver, gmres_iter);
         HYPRE_GMRESSetTol(pcg_solver, 0.0);
         HYPRE_GMRESSolve(pcg_solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)b, 
                       (HYPRE_Vector)x);
         HYPRE_GMRESGetNumIterations(pcg_solver, &num_iterations);
         HYPRE_GMRESGetFinalRelativeResidualNorm(pcg_solver,&final_res_norm);
         if (myid == 0)
         {
            hypre_printf("\n");
            hypre_printf("GMRES Iterations = %d\n", num_iterations);
            hypre_printf("Final GMRES Relative Residual Norm = %e\n", final_res_norm);
            hypre_printf("\n");
         }
       }
 

       HYPRE_ParCSRGMRESDestroy(pcg_solver);
 
       if (solver_id == 3 )
       {
         HYPRE_BoomerAMGDestroy(pcg_precond);
       }

       if (j < time_steps) AddOrRestoreAIJ(ij_A, 26.0, 0);
     }
   }
   hypre_EndTiming(time_index);
   hypre_PrintTiming("Solve phase times", hypre_MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
 

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   if (print_system)
   {
      HYPRE_IJVectorPrint(ij_x, "IJ.out.x");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   if (build_matrix_type == -1) HYPRE_IJMatrixDestroy(ij_A);
   else HYPRE_ParCSRMatrixDestroy(parcsr_A);

   /* for build_rhs_type = 1 or 7, we did not create ij_b  - just b*/
   if (build_rhs_type ==1 || build_rhs_type ==7 || build_rhs_type==6)
      HYPRE_ParVectorDestroy(b);
   else
      HYPRE_IJVectorDestroy(ij_b);

   HYPRE_IJVectorDestroy(ij_x);

/*
  hypre_FinalizeMemoryDebug();
*/
   hypre_GPUFinalize();
   hypre_MPI_Finalize();

   return (0);
}



/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian( HYPRE_Int                  argc,
                   char                *argv[],
                   HYPRE_Int                  arg_index,
                   HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;
   HYPRE_Real          cx, cy, cz;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Real         *values;

   HYPRE_Int                 system_vcoef = 0;
   HYPRE_Int                 sys_opt = 0;
   HYPRE_Int                 vcoef_opt = 0;
   
   
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("  Laplacian:   \n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(HYPRE_Real, 4);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian(hypre_MPI_COMM_WORLD, 
                                                 nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point convection-diffusion operator 
 * Parameters given in command line.
 * Operator:
 *
 *  -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f
 *
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParDifConv( HYPRE_Int                  argc,
                 char                *argv[],
                 HYPRE_Int                  arg_index,
                 HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;
   HYPRE_Real          cx, cy, cz;
   HYPRE_Real          ax, ay, az;
   HYPRE_Real          hinx,hiny,hinz;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   ax = 1.;
   ay = 1.;
   az = 1.;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-a") == 0 )
      {
         arg_index++;
         ax = atof(argv[arg_index++]);
         ay = atof(argv[arg_index++]);
         az = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("  Convection-Diffusion: \n");
      hypre_printf("    -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f\n");  
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      hypre_printf("    (ax, ay, az) = (%f, %f, %f)\n\n", ax, ay, az);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   hinx = 1./(nx+1);
   hiny = 1./(ny+1);
   hinz = 1./(nz+1);

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(HYPRE_Real, 7);

   values[1] = -cx/(hinx*hinx);
   values[2] = -cy/(hiny*hiny);
   values[3] = -cz/(hinz*hinz);
   values[4] = -cx/(hinx*hinx) + ax/hinx;
   values[5] = -cy/(hiny*hiny) + ay/hiny;
   values[6] = -cz/(hinz*hinz) + az/hinz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx/(hinx*hinx) - 1.*ax/hinx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy/(hiny*hiny) - 1.*ay/hiny;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz/(hinz*hinz) - 1.*az/hinz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateDifConv(hypre_MPI_COMM_WORLD,
                                            nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParLaplacian27pt( HYPRE_Int                  argc,
                       char                *argv[],
                       HYPRE_Int                  arg_index,
                       HYPRE_ParCSRMatrix  *A_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      hypre_printf("  Laplacian_27pt:\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(HYPRE_Real, 2);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
      values[0] = 8.0;
   if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
      values[0] = 2.0;
   values[1] = -1.;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   hypre_TFree(values);

   *A_ptr = A;

   return (0);
}


/*----------------------------------------------------------------------
 * Build standard 7-point difference operator using centered differences
 *
 *  eps*(a(x,y,z) ux)x + (b(x,y,z) uy)y + (c(x,y,z) uz)z 
 *  d(x,y,z) ux + e(x,y,z) uy + f(x,y,z) uz + g(x,y,z) u
 *
 *  functions a,b,c,d,e,f,g need to be defined inside par_vardifconv.c
 *
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildParVarDifConv( HYPRE_Int                  argc,
                    char                *argv[],
                    HYPRE_Int                  arg_index,
                    HYPRE_ParCSRMatrix  *A_ptr    ,
                    HYPRE_ParVector  *rhs_ptr     )
{
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;
   HYPRE_ParVector  rhs;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q, r;
   HYPRE_Real          eps;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;
   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-eps") == 0 )
      {
         arg_index++;
         eps  = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  ell PDE: eps = %f\n", eps);
      hypre_printf("    Dx(aDxu) + Dy(bDyu) + Dz(cDzu) + d Dxu + e Dyu + f Dzu  + g u= f\n");
      hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
   }
   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   A = (HYPRE_ParCSRMatrix) GenerateVarDifConv(hypre_MPI_COMM_WORLD,
                                               nx, ny, nz, P, Q, R, p, q, r, eps, &rhs);

   *A_ptr = A;
   *rhs_ptr = rhs;

   return (0);
}

/**************************************************************************/



/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int
BuildIJLaplacian27pt( HYPRE_Int                  argc,
                       char                *argv[],
                       HYPRE_Int                  arg_index,
                       HYPRE_IJMatrix  *ij_A_ptr     )
{
   MPI_Comm            comm = hypre_MPI_COMM_WORLD;
   HYPRE_Int                 nx, ny, nz;
   HYPRE_Int                 P, Q, R;

   HYPRE_IJMatrix  ij_A;

   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 px, py, pz;
   HYPRE_Real         *value;
   HYPRE_Int    *diag_i;
   HYPRE_Int    *offd_i;
   HYPRE_Int    *row_nums;
   HYPRE_Int    *col_nums;
   HYPRE_Int    *num_cols;
   HYPRE_Real *data;

   HYPRE_Int row_index;
   HYPRE_Int i;
   HYPRE_Int local_size;
   HYPRE_Int first_local_row;
   HYPRE_Int last_local_row;
   HYPRE_Int first_local_col;
   HYPRE_Int last_local_col;

   HYPRE_Int nxy;
   HYPRE_Int nx_global, ny_global, nz_global;
   HYPRE_Int all_threads;
   HYPRE_Int *nnz;
   HYPRE_Int Cx, Cy, Cz;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs );
   hypre_MPI_Comm_rank(comm, &myid );
   all_threads = hypre_NumThreads();
   nnz = hypre_CTAlloc(HYPRE_Int, all_threads);

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
 
   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   nx_global = P*nx; 
   ny_global = Q*ny; 
   nz_global = R*nz; 
   if (myid == 0)

   {
      hypre_printf("  Laplacian_27pt:\n");
      hypre_printf("    (Nx, Ny, Nz) = (%d, %d, %d)\n", nx_global, ny_global, nz_global);
      hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute px,py,pz from P,Q,R and myid */
   px = myid % P;
   py = (( myid - px)/P) % Q;
   pz = ( myid - px - P*py)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   value = hypre_CTAlloc(HYPRE_Real, 2);

   value[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
      value[0] = 8.0;
   if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
      value[0] = 2.0;
   value[1] = -1.;

   local_size = nx*ny*nz;

   row_nums = hypre_CTAlloc(HYPRE_Int, local_size);
   num_cols = hypre_CTAlloc(HYPRE_Int, local_size);
   row_index = myid*local_size;

   HYPRE_IJMatrixCreate( comm, row_index, row_index+local_size-1,
                               row_index, row_index+local_size-1,
                               &ij_A );

   HYPRE_IJMatrixSetObjectType( ij_A, HYPRE_PARCSR );

   nxy = nx*ny;

   diag_i = hypre_CTAlloc(HYPRE_Int, local_size);
   offd_i = hypre_CTAlloc(HYPRE_Int, local_size);

   Cx = nx*(ny*nz-1);
   Cy = nxy*(P*nz-1);
   Cz = local_size*(P*Q-1);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel
#endif
   {
    HYPRE_Int ix, iy, iz;
    HYPRE_Int cnt, o_cnt;
    HYPRE_Int ix_start, ix_end;
    HYPRE_Int iy_start, iy_end;
    HYPRE_Int iz_start, iz_end;
    HYPRE_Int num_threads, my_thread;
    HYPRE_Int all_nnz=0;
    HYPRE_Int size, rest;
    HYPRE_Int new_row_index;
    num_threads = hypre_NumActiveThreads();
    my_thread = hypre_GetThreadNum();
    size = nz/num_threads;
    rest = nz - size*num_threads;
    ix_start = nx*px;
    ix_end = ix_start+nx;
    iy_start = ny*py;
    iy_end = iy_start+ny;
    if (my_thread < rest)
    {
       iz_start = nz*pz + my_thread*size+my_thread;
       iz_end = nz*pz + (my_thread+1)*size+my_thread+1;
       cnt = (my_thread*size+my_thread)*nxy-1;
    }
    else
    {
       iz_start = nz*pz + my_thread*size+rest;
       iz_end = nz*pz + (my_thread+1)*size+rest;
       cnt = (my_thread*size+rest)*nxy-1;
    }
    o_cnt = cnt;

    for (iz = iz_start;  iz < iz_end; iz++)
    {
      for (iy = iy_start;  iy < iy_end; iy++)
      {
         for (ix = ix_start; ix < ix_end; ix++)
         {
            cnt++;
            o_cnt++;
            diag_i[cnt]++;
            if (iz > nz*pz) 
            {
               diag_i[cnt]++;
               if (iy > ny*py) 
               {
                  diag_i[cnt]++;
      	          if (ix > nx*px)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix) 
      		        offd_i[o_cnt]++;
      	          }
      	          if (ix < nx*(px+1)-1)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
      		        offd_i[o_cnt]++;
      	          }
               }
               else
               {
                  if (iy) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx*px)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             if (ix < nx*(px+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix < nx_global-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
                  }
               }
               if (ix > nx*px) 
                  diag_i[cnt]++;
               else
               {
                  if (ix) 
                  {
                     offd_i[o_cnt]++;
                  }
               }
               if (ix+1 < nx*(px+1)) 
                  diag_i[cnt]++;
               else
               {
                  if (ix+1 < nx_global) 
                  {
                     offd_i[o_cnt]++; 
                  }
               }
               if (iy+1 < ny*(py+1)) 
               {
                  diag_i[cnt]++;
      	          if (ix > nx*px)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix) 
      		        offd_i[o_cnt]++;
      	          }
      	          if (ix < nx*(px+1)-1)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
      		        offd_i[o_cnt]++;
      	          }
               }
               else
               {
                  if (iy+1 < ny_global) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx*px)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             if (ix < nx*(px+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix < nx_global-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
                  }
               }
            }
            else
            {
               if (iz)
	       {
		  offd_i[o_cnt]++;
                  if (iy > ny*py) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx*px)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix) 
      	   	        offd_i[o_cnt]++;
      	             }
      	             if (ix < nx*(px+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	   	        offd_i[o_cnt]++;
      	             }
                  }
                  else
                  {
                     if (iy) 
                     {
                        offd_i[o_cnt]++;
      	                if (ix > nx*px)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                if (ix < nx*(px+1)-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix < nx_global-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
                     }
                  }
                  if (ix > nx*px) 
                     offd_i[o_cnt]++;
                  else
                  {
                     if (ix) 
                     {
                        offd_i[o_cnt]++; 
                     }
                  }
                  if (ix+1 < nx*(px+1)) 
                     offd_i[o_cnt]++;
                  else
                  {
                     if (ix+1 < nx_global) 
                     {
                        offd_i[o_cnt]++; 
                     }
                  }
                  if (iy+1 < ny*(py+1)) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx*px)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix) 
      	   	           offd_i[o_cnt]++;
      	             }
      	             if (ix < nx*(px+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	   	           offd_i[o_cnt]++;
      	             }
                  }
                  else
                  {
                     if (iy+1 < ny_global) 
                     {
                        offd_i[o_cnt]++;
      	                if (ix > nx*px)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                if (ix < nx*(px+1)-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix < nx_global-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
                     }
                  }
               }
            }
            if (iy > ny*py) 
            {
               diag_i[cnt]++;
   	       if (ix > nx*px)
   	       {
   	          diag_i[cnt]++;
   	       }
   	       else
   	       {
   	          if (ix) 
   		     offd_i[o_cnt]++;
   	       }
   	       if (ix < nx*(px+1)-1)
   	       {
   	          diag_i[cnt]++;
   	       }
   	       else
   	       {
   	          if (ix+1 < nx_global) 
   		     offd_i[o_cnt]++;
   	       }
            }
            else
            {
               if (iy) 
               {
                  offd_i[o_cnt]++;
   	          if (ix > nx*px)
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          else if (ix)
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          if (ix < nx*(px+1)-1)
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          else if (ix < nx_global-1)
   	          {
   	             offd_i[o_cnt]++;
   	          }
               }
            }
            if (ix > nx*px) 
               diag_i[cnt]++;
            else
            {
               if (ix) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (ix+1 < nx*(px+1)) 
               diag_i[cnt]++;
            else
            {
               if (ix+1 < nx_global) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (iy+1 < ny*(py+1)) 
            {
               diag_i[cnt]++;
   	       if (ix > nx*px)
   	       {
   	          diag_i[cnt]++;
   	       }
   	       else
   	       {
   	          if (ix) 
   		     offd_i[o_cnt]++;
   	       }
   	       if (ix < nx*(px+1)-1)
   	       {
   	          diag_i[cnt]++;
   	       }
   	       else
   	       {
   	          if (ix+1 < nx_global) 
   		     offd_i[o_cnt]++;
   	       }
            }
            else
            {
               if (iy+1 < ny_global) 
               {
                  offd_i[o_cnt]++;
   	          if (ix > nx*px)
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          else if (ix)
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          if (ix < nx*(px+1)-1)
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          else if (ix < nx_global-1)
   	          {
   	             offd_i[o_cnt]++;
   	          }
               }
            }
            if (iz+1 < nz*(pz+1)) 
            {
               diag_i[cnt]++;
               if (iy > ny*py) 
               {
                  diag_i[cnt]++;
      	          if (ix > nx*px)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix) 
      		        offd_i[o_cnt]++;
      	          }
      	          if (ix < nx*(px+1)-1)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
      		        offd_i[o_cnt]++;
      	          }
               }
               else
               {
                  if (iy) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx*px)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             if (ix < nx*(px+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix < nx_global-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
                  }
               }
               if (ix > nx*px) 
                  diag_i[cnt]++;
               else
               {
                  if (ix) 
                  {
                     offd_i[o_cnt]++; 
                  }
               }
               if (ix+1 < nx*(px+1)) 
                  diag_i[cnt]++;
               else
               {
                  if (ix+1 < nx_global) 
                  {
                     offd_i[o_cnt]++; 
                  }
               }
               if (iy+1 < ny*(py+1)) 
               {
                  diag_i[cnt]++;
      	          if (ix > nx*px)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix) 
      		        offd_i[o_cnt]++;
      	          }
      	          if (ix < nx*(px+1)-1)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
      		        offd_i[o_cnt]++;
      	          }
               }
               else
               {
                  if (iy+1 < ny_global) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx*px)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             if (ix < nx*(px+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix < nx_global-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
                  }
               }
            }
            else
            {
               if (iz+1 < nz_global)
	       {
		  offd_i[o_cnt]++;
                  if (iy > ny*py) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx*px)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix) 
      	   	        offd_i[o_cnt]++;
      	             }
      	             if (ix < nx*(px+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	   	        offd_i[o_cnt]++;
      	             }
                  }
                  else
                  {
                     if (iy) 
                     {
                        offd_i[o_cnt]++;
      	                if (ix > nx*px)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                if (ix < nx*(px+1)-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix < nx_global-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
                     }
                  }
                  if (ix > nx*px) 
                     offd_i[o_cnt]++;
                  else
                  {
                     if (ix) 
                     {
                        offd_i[o_cnt]++; 
                     }
                  }
                  if (ix+1 < nx*(px+1)) 
                     offd_i[o_cnt]++;
                  else
                  {
                     if (ix+1 < nx_global) 
                     {
                        offd_i[o_cnt]++; 
                     }
                  }
                  if (iy+1 < ny*(py+1)) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx*px)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix) 
      	   	           offd_i[o_cnt]++;
      	             }
      	             if (ix < nx*(px+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	   	           offd_i[o_cnt]++;
      	             }
                  }
                  else
                  {
                     if (iy+1 < ny_global) 
                     {
                        offd_i[o_cnt]++;
      	                if (ix > nx*px)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                if (ix < nx*(px+1)-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix < nx_global-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
                     }
                  }
               }
            }
            nnz[my_thread] += diag_i[cnt]+offd_i[o_cnt];
            row_nums[cnt] = row_index+cnt;
            num_cols[cnt] = diag_i[cnt]+offd_i[o_cnt];
         }
      }
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

   if (my_thread == 0)
   {
      for (i=1; i< num_threads; i++)
         nnz[i]+= nnz[i-1];

      all_nnz = nnz[num_threads-1];
      col_nums = hypre_CTAlloc(HYPRE_Int, all_nnz);
      data = hypre_CTAlloc(HYPRE_Real, all_nnz);

      HYPRE_IJMatrixSetDiagOffdSizes( ij_A, diag_i, offd_i);
   }

#ifdef HYPRE_USING_OPENMP
#pragma omp barrier
#endif

   if (my_thread) 
   {
      cnt = nnz[my_thread-1];
      new_row_index = row_index+(iz_start-nz*pz)*nxy;
   }
   else
   {
      cnt = 0;
      new_row_index = row_index;
   }
   for (iz = iz_start;  iz < iz_end; iz++)
   {
      for (iy = iy_start;  iy < iy_end; iy++)
      {
         for (ix = ix_start; ix < ix_end; ix++)
         {
            col_nums[cnt] = new_row_index;
            data[cnt++] = value[0];
            if (iz > nz*pz) 
            {
               if (iy > ny*py) 
               {
      	          if (ix > nx*px)
      	          {
      	             col_nums[cnt] = new_row_index-nxy-nx-1;
      	             data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix) 
      	             {
      		        col_nums[cnt] = hypre_map27(ix-1,iy-1,iz-1,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	          }
      	          col_nums[cnt] = new_row_index-nxy-nx;
      	          data[cnt++] = value[1];
      	          if (ix < nx*(px+1)-1)
      	          {
      	             col_nums[cnt] = new_row_index-nxy-nx+1;
      	             data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
      	             {
      		        col_nums[cnt] = hypre_map27(ix+1,iy-1,iz-1,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	          }
               }
               else
               {
                  if (iy) 
                  {
      	             if (ix > nx*px)
      	             {
      		        col_nums[cnt] = hypre_map27(ix-1,iy-1,iz-1,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix)
      	             {
      		        col_nums[cnt] = hypre_map27(ix-1,iy-1,iz-1,px-1,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      		     col_nums[cnt] = hypre_map27(ix,iy-1,iz-1,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < nx*(px+1)-1)
      	             {
      		        col_nums[cnt] = hypre_map27(ix+1,iy-1,iz-1,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix < nx_global-1)
      	             {
      		        col_nums[cnt] = hypre_map27(ix+1,iy-1,iz-1,px+1,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
                  }
               }
               if (ix > nx*px) 
      	       {   
      	          col_nums[cnt] = new_row_index-nxy-1;
      	          data[cnt++] = value[1];
      	       }   
               else
               {
                  if (ix) 
                  {
      		     col_nums[cnt] = hypre_map27(ix-1,iy,iz-1,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
               }
      	       col_nums[cnt] = new_row_index-nxy;
      	       data[cnt++] = value[1];
               if (ix+1 < nx*(px+1)) 
      	       {   
      	          col_nums[cnt] = new_row_index-nxy+1;
      	          data[cnt++] = value[1];
      	       }   
               else
               {
                  if (ix+1 < nx_global) 
                  {
      		     col_nums[cnt] = hypre_map27(ix+1,iy,iz-1,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
               }
               if (iy+1 < ny*(py+1)) 
               {
      	          if (ix > nx*px)
      	          {
      	             col_nums[cnt] = new_row_index-nxy+nx-1;
      	             data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix) 
                     {
      		        col_nums[cnt] = hypre_map27(ix-1,iy+1,iz-1,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
      	          }
      	          col_nums[cnt] = new_row_index-nxy+nx;
      	          data[cnt++] = value[1];
      	          if (ix < nx*(px+1)-1)
      	          {
      	             col_nums[cnt] = new_row_index-nxy+nx+1;
      	             data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
                     {
      		        col_nums[cnt] = hypre_map27(ix+1,iy+1,iz-1,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
      	          }
               }
               else
               {
                  if (iy+1 < ny_global) 
                  {
      	             if (ix > nx*px)
      	             {
      		        col_nums[cnt] = hypre_map27(ix-1,iy+1,iz-1,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix)
      	             {
      		        col_nums[cnt] = hypre_map27(ix-1,iy+1,iz-1,px-1,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      		     col_nums[cnt] = hypre_map27(ix,iy+1,iz-1,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < nx*(px+1)-1)
      	             {
      		        col_nums[cnt] = hypre_map27(ix+1,iy+1,iz-1,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix < nx_global-1)
      	             {
      		        col_nums[cnt] = hypre_map27(ix+1,iy+1,iz-1,px+1,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
                  }
               }
            }
            else
            {
               if (iz)
	       {
                  if (iy > ny*py) 
                  {
      	             if (ix > nx*px)
      	             {
      		        col_nums[cnt] = hypre_map27(ix-1,iy-1,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix) 
      	                {
      		           col_nums[cnt] = hypre_map27(ix-1,iy-1,iz-1,px-1,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
      		     col_nums[cnt] = hypre_map27(ix,iy-1,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < nx*(px+1)-1)
      	             {
      		        col_nums[cnt] = hypre_map27(ix+1,iy-1,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	                {
      		           col_nums[cnt] = hypre_map27(ix+1,iy-1,iz-1,px+1,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
                  }
                  else
                  {
                     if (iy) 
                     {
      	                if (ix > nx*px)
      	                {
      		           col_nums[cnt] = hypre_map27(ix-1,iy-1,iz-1,px,py-1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix)
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy-1,iz-1,px-1,py-1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      		        col_nums[cnt] = hypre_map27(ix,iy-1,iz-1,px,py-1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	                if (ix < nx*(px+1)-1)
      	                {
      		           col_nums[cnt] = hypre_map27(ix+1,iy-1,iz-1,px,py-1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix < nx_global-1)
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy-1,iz-1,px+1,py-1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
                     }
                  }
                  if (ix > nx*px) 
                  {
      		     col_nums[cnt] =hypre_map27(ix-1,iy,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix) 
                     {
      		        col_nums[cnt] =hypre_map27(ix-1,iy,iz-1,px-1,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
                  }
      		  col_nums[cnt] =hypre_map27(ix,iy,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		  data[cnt++] = value[1];
                  if (ix+1 < nx*(px+1)) 
                  {
      		     col_nums[cnt] =hypre_map27(ix+1,iy,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix+1 < nx_global) 
                     {
      		        col_nums[cnt] =hypre_map27(ix+1,iy,iz-1,px+1,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
                  }
                  if (iy+1 < ny*(py+1)) 
                  {
      	             if (ix > nx*px)
      	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy+1,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix) 
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy+1,iz-1,px-1,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
      		     col_nums[cnt] =hypre_map27(ix,iy+1,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < nx*(px+1)-1)
      	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy+1,iz-1,px,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy+1,iz-1,px+1,py,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
                  }
                  else
                  {
                     if (iy+1 < ny_global) 
                     {
      	                if (ix > nx*px)
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy+1,iz-1,px,py+1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix)
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy+1,iz-1,px-1,py+1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      		        col_nums[cnt] =hypre_map27(ix,iy+1,iz-1,px,py+1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	                if (ix < nx*(px+1)-1)
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy+1,iz-1,px,py+1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix < nx_global-1)
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy+1,iz-1,px+1,py+1,pz-1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
                     }
                  }
               }
            }
            if (iy > ny*py) 
            {
   	       if (ix > nx*px)
   	       {
   	          col_nums[cnt] = new_row_index-nx-1;
   	          data[cnt++] = value[1];
   	       }
   	       else
   	       {
   	          if (ix) 
   	          {
      		     col_nums[cnt] =hypre_map27(ix-1,iy-1,iz,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
   	       }
   	       col_nums[cnt] = new_row_index-nx;
   	       data[cnt++] = value[1];
   	       if (ix < nx*(px+1)-1)
   	       {
   	          col_nums[cnt] = new_row_index-nx+1;
   	          data[cnt++] = value[1];
   	       }
   	       else
   	       {
   	          if (ix+1 < nx_global) 
   	          {
      		     col_nums[cnt] =hypre_map27(ix+1,iy-1,iz,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
   	       }
            }
            else
            {
               if (iy) 
               {
   	          if (ix > nx*px)
   	          {
      		     col_nums[cnt] =hypre_map27(ix-1,iy-1,iz,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
   	          else if (ix)
   	          {
      		     col_nums[cnt] =hypre_map27(ix-1,iy-1,iz,px-1,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
      		  col_nums[cnt] =hypre_map27(ix,iy-1,iz,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		  data[cnt++] = value[1];
   	          if (ix < nx*(px+1)-1)
   	          {
      		     col_nums[cnt] =hypre_map27(ix+1,iy-1,iz,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
   	          else if (ix < nx_global-1)
   	          {
      		     col_nums[cnt] =hypre_map27(ix+1,iy-1,iz,px+1,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
               }
            }
            if (ix > nx*px) 
            {
               col_nums[cnt] = new_row_index-1;
               data[cnt++] = value[1];
            }
            else
            {
               if (ix) 
               {
      		  col_nums[cnt] =hypre_map27(ix-1,iy,iz,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		  data[cnt++] = value[1];
               }
            }
            if (ix+1 < nx*(px+1)) 
            {
               col_nums[cnt] = new_row_index+1;
               data[cnt++] = value[1];
            }
            else
            {
               if (ix+1 < nx_global) 
               {
      		  col_nums[cnt] =hypre_map27(ix+1,iy,iz,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		  data[cnt++] = value[1];
               }
            }
            if (iy+1 < ny*(py+1)) 
            {
   	       if (ix > nx*px)
   	       {
                  col_nums[cnt] = new_row_index+nx-1;
                  data[cnt++] = value[1];
   	       }
   	       else
   	       {
   	          if (ix) 
                  {
      		     col_nums[cnt] =hypre_map27(ix-1,iy+1,iz,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
   	       }
               col_nums[cnt] = new_row_index+nx;
               data[cnt++] = value[1];
   	       if (ix < nx*(px+1)-1)
   	       {
                  col_nums[cnt] = new_row_index+nx+1;
                  data[cnt++] = value[1];
   	       }
   	       else
   	       {
   	          if (ix+1 < nx_global) 
                  {
      		     col_nums[cnt] =hypre_map27(ix+1,iy+1,iz,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
   	       }
            }
            else
            {
               if (iy+1 < ny_global) 
               {
   	          if (ix > nx*px)
   	          {
      		     col_nums[cnt] =hypre_map27(ix-1,iy+1,iz,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
   	          else if (ix)
   	          {
      		     col_nums[cnt] =hypre_map27(ix-1,iy+1,iz,px-1,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
      		  col_nums[cnt] =hypre_map27(ix,iy+1,iz,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		  data[cnt++] = value[1];
   	          if (ix < nx*(px+1)-1)
   	          {
      		     col_nums[cnt] =hypre_map27(ix+1,iy+1,iz,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
   	          else if (ix < nx_global-1)
   	          {
      		     col_nums[cnt] =hypre_map27(ix+1,iy+1,iz,px+1,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
   	          }
               }
            }
            if (iz+1 < nz*(pz+1)) 
            {
               if (iy > ny*py) 
               {
      	          if (ix > nx*px)
      	          {
      	             col_nums[cnt] = new_row_index+nxy-nx-1;
      	             data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix) 
   	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy-1,iz+1,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
   	             }
      	          }
      	          col_nums[cnt] = new_row_index+nxy-nx;
      	          data[cnt++] = value[1];
      	          if (ix < nx*(px+1)-1)
      	          {
      	             col_nums[cnt] = new_row_index+nxy-nx+1;
      	             data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
   	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy-1,iz+1,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
   	             }
      	          }
               }
               else
               {
                  if (iy) 
                  {
      	             if (ix > nx*px)
      	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy-1,iz+1,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix)
      	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy-1,iz+1,px-1,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      		     col_nums[cnt] =hypre_map27(ix,iy-1,iz+1,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < nx*(px+1)-1)
      	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy-1,iz+1,px,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix < nx_global-1)
      	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy-1,iz+1,px+1,py-1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
                  }
               }
               if (ix > nx*px) 
               {
                  col_nums[cnt] = new_row_index+nxy-1;
                  data[cnt++] = value[1];
               }
               else
               {
                  if (ix) 
                  {
      		     col_nums[cnt] =hypre_map27(ix-1,iy,iz+1,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
               }
               col_nums[cnt] = new_row_index+nxy;
               data[cnt++] = value[1];
               if (ix+1 < nx*(px+1)) 
               {
                  col_nums[cnt] = new_row_index+nxy+1;
                  data[cnt++] = value[1];
               }
               else
               {
                  if (ix+1 < nx_global) 
                  {
      		     col_nums[cnt] =hypre_map27(ix+1,iy,iz+1,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
               }
               if (iy+1 < ny*(py+1)) 
               {
      	          if (ix > nx*px)
      	          {
                     col_nums[cnt] = new_row_index+nxy+nx-1;
                     data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix) 
                     {
      		        col_nums[cnt] =hypre_map27(ix-1,iy+1,iz+1,px-1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
      	          }
                  col_nums[cnt] = new_row_index+nxy+nx;
                  data[cnt++] = value[1];
      	          if (ix < nx*(px+1)-1)
      	          {
                     col_nums[cnt] = new_row_index+nxy+nx+1;
                     data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix+1 < nx_global) 
                     {
      		        col_nums[cnt] =hypre_map27(ix+1,iy+1,iz+1,px+1,py,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
      	          }
               }
               else
               {
                  if (iy+1 < ny_global) 
                  {
      	             if (ix > nx*px)
      	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy+1,iz+1,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix)
      	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy+1,iz+1,px-1,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      		     col_nums[cnt] =hypre_map27(ix,iy+1,iz+1,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < nx*(px+1)-1)
      	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy+1,iz+1,px,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else if (ix < nx_global-1)
      	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy+1,iz+1,px+1,py+1,pz,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
                  }
               }
            }
            else
            {
               if (iz+1 < nz_global)
	       {
                  if (iy > ny*py) 
                  {
      	             if (ix > nx*px)
      	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy-1,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix) 
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy-1,iz+1,px-1,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
      		     col_nums[cnt] =hypre_map27(ix,iy-1,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < nx*(px+1)-1)
      	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy-1,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy-1,iz+1,px+1,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
                  }
                  else
                  {
                     if (iy) 
                     {
      	                if (ix > nx*px)
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy-1,iz+1,px,py-1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix)
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy-1,iz+1,px-1,py-1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      		        col_nums[cnt] =hypre_map27(ix,iy-1,iz+1,px,py-1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	                if (ix < nx*(px+1)-1)
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy-1,iz+1,px,py-1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix < nx_global-1)
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy-1,iz+1,px+1,py-1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
                     }
                  }
                  if (ix > nx*px) 
                  {
      		     col_nums[cnt] =hypre_map27(ix-1,iy,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix) 
                     {
      		        col_nums[cnt] =hypre_map27(ix-1,iy,iz+1,px-1,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
                  }
      		  col_nums[cnt] =hypre_map27(ix,iy,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		  data[cnt++] = value[1];
                  if (ix+1 < nx*(px+1)) 
                  {
      		     col_nums[cnt] =hypre_map27(ix+1,iy,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
                  }
                  else
                  {
                     if (ix+1 < nx_global) 
                     {
      		        col_nums[cnt] =hypre_map27(ix+1,iy,iz+1,px+1,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
                     }
                  }
                  if (iy+1 < ny*(py+1)) 
                  {
      	             if (ix > nx*px)
      	             {
      		        col_nums[cnt] =hypre_map27(ix-1,iy+1,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix) 
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy+1,iz+1,px-1,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
      		     col_nums[cnt] =hypre_map27(ix,iy+1,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		     data[cnt++] = value[1];
      	             if (ix < nx*(px+1)-1)
      	             {
      		        col_nums[cnt] =hypre_map27(ix+1,iy+1,iz+1,px,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix+1 < nx_global) 
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy+1,iz+1,px+1,py,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	             }
                  }
                  else
                  {
                     if (iy+1 < ny_global) 
                     {
      	                if (ix > nx*px)
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy+1,iz+1,px,py+1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix)
      	                {
      		           col_nums[cnt] =hypre_map27(ix-1,iy+1,iz+1,px-1,py+1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      		        col_nums[cnt] =hypre_map27(ix,iy+1,iz+1,px,py+1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		        data[cnt++] = value[1];
      	                if (ix < nx*(px+1)-1)
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy+1,iz+1,px,py+1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
      	                else if (ix < nx_global-1)
      	                {
      		           col_nums[cnt] =hypre_map27(ix+1,iy+1,iz+1,px+1,py+1,pz+1,
					Cx,Cy,Cz,nx,nxy);
      		           data[cnt++] = value[1];
      	                }
                     }
                  }
               }
            }
            new_row_index++;
         }
      }
    }
   } /*end parallel loop */

   HYPRE_IJMatrixInitialize(ij_A);

   HYPRE_IJMatrixSetOMPFlag(ij_A, 1);

   HYPRE_IJMatrixSetValues(ij_A, local_size, num_cols, row_nums,
			col_nums, data);

   HYPRE_IJMatrixAssemble(ij_A);

   /*A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, px, py, r, values);*/

   hypre_TFree(diag_i);
   hypre_TFree(offd_i);
   hypre_TFree(num_cols);
   hypre_TFree(col_nums);
   hypre_TFree(row_nums);
   hypre_TFree(data);
   hypre_TFree(value);
   hypre_TFree(nnz);

   *ij_A_ptr = ij_A;

   return (0);
}

/*----------------------------------------------------------------------
 * Add epsilon to diagonal of A
 *----------------------------------------------------------------------*/

HYPRE_Int
AddOrRestoreAIJ( HYPRE_IJMatrix  ij_A, HYPRE_Real eps, HYPRE_Int action  )
{
   HYPRE_Int first_row, last_row, i;
   HYPRE_Int first_col, last_col, local_size;
   HYPRE_Int *num_cols;
   HYPRE_Int *row_nums;
   HYPRE_Int *col_nums;
   HYPRE_Real *data;
   HYPRE_Real diagonal = 26.0;

   if (action) diagonal = eps;
  
   HYPRE_IJMatrixGetLocalRange(ij_A, &first_row, &last_row, &first_col, &last_col);

   local_size = last_row-first_row+1;
   num_cols = hypre_CTAlloc(HYPRE_Int, local_size);
   row_nums = hypre_CTAlloc(HYPRE_Int, local_size);
   col_nums = hypre_CTAlloc(HYPRE_Int, local_size);
   data = hypre_CTAlloc(HYPRE_Real, local_size);

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i=0; i < local_size; i++)
   {
      num_cols[i] = 1;
      row_nums[i] = first_row+i;
      col_nums[i] = first_row+i;
      data[i] = diagonal;
   }

   if (action) 
      HYPRE_IJMatrixAddToValues(ij_A, local_size, num_cols, row_nums,
			col_nums, data);
   else
      HYPRE_IJMatrixSetValues(ij_A, local_size, num_cols, row_nums,
			col_nums, data);

   HYPRE_IJMatrixAssemble(ij_A);
   return(0);
}

HYPRE_Int
hypre_map27( HYPRE_Int  ix,
      HYPRE_Int  iy,
      HYPRE_Int  iz,
      HYPRE_Int  px,
      HYPRE_Int  py,
      HYPRE_Int  pz,
      HYPRE_Int  Cx,
      HYPRE_Int  Cy,
      HYPRE_Int  Cz,
      HYPRE_Int nx,
      HYPRE_Int nxy)
{
   HYPRE_Int global_index = pz*Cz + py*Cy +px*Cx + iz*nxy + iy*nx + ix;

   return global_index;
}
