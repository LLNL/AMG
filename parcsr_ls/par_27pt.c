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




 
#include "_hypre_parcsr_ls.h"
 
/*--------------------------------------------------------------------------
 * hypre_GenerateLaplacian27pt
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix 
GenerateLaplacian27pt(MPI_Comm comm,
                      HYPRE_Int      nx_local,
                      HYPRE_Int      ny_local,
                      HYPRE_Int      nz_local,
                      HYPRE_Int      P,
                      HYPRE_Int      Q,
                      HYPRE_Int      R,
                      HYPRE_Int      p,
                      HYPRE_Int      q,
                      HYPRE_Int      r,
                      HYPRE_Real  *value )
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;

   HYPRE_Int    *diag_i;
   HYPRE_Int    *diag_j;
   HYPRE_Real *diag_data;

   HYPRE_Int    *offd_i;
   HYPRE_Int    *offd_j;
   HYPRE_Real *offd_data;

   HYPRE_Int ix, iy, iz;
   HYPRE_Int cnt, o_cnt;
   HYPRE_Int local_num_rows; 
   HYPRE_Int *col_map_offd = NULL;
   HYPRE_Int *work;
   HYPRE_Int row_index;
   HYPRE_Int i, j;
   HYPRE_Int *global_part = NULL;

   HYPRE_Int nx, ny, nz;
   HYPRE_Int nx_size, ny_size, nz_size;
   HYPRE_Int num_cols_offd;
   HYPRE_Int nxy;
   HYPRE_Int grid_size;

   HYPRE_Int num_procs, my_id;

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

   my_id = (r*Q + q)*P + p;
   num_procs = P*Q*R;
   local_num_rows = nx_local*ny_local*nz_local;
   grid_size = num_procs*local_num_rows;
   nx = nx_local*P;
   ny = ny_local*Q;
   nz = nz_local*R;

   local_num_rows = nx_local*ny_local*nz_local;
   diag_i = hypre_CTAlloc(HYPRE_Int, local_num_rows+1);
   offd_i = hypre_CTAlloc(HYPRE_Int, local_num_rows+1);

   num_cols_offd = 0;
   if (p) 
   {
      num_cols_offd += ny_local*nz_local;
      if (q) 
      {
	 num_cols_offd += nz_local;
         if ( r) num_cols_offd++;
         if (r < R-1) num_cols_offd++;
      }
      if (q < Q-1 )
      {
         num_cols_offd += nz_local;
         if (r) num_cols_offd++;
         if (r < R-1) num_cols_offd++;
      }
      if (r) num_cols_offd += ny_local;
      if (r < R-1 ) num_cols_offd += ny_local;
   }
   if (p < P-1) 
   {
      num_cols_offd += ny_local*nz_local;
      if (q) 
      {
         num_cols_offd += nz_local;
         if ( r) num_cols_offd++;
         if ( r < R-1 ) num_cols_offd++;
      }
      if (q < Q-1 )
      {
         num_cols_offd += nz_local;
         if ( r ) num_cols_offd++;
         if ( r < R-1) num_cols_offd++;
      }
      if ( r ) num_cols_offd += ny_local;
      if ( r < R-1 ) num_cols_offd += ny_local;
   }
   if (q)
   {
      num_cols_offd += nx_local*nz_local;
      if (r) num_cols_offd += nx_local;
      if ( r < R-1 ) num_cols_offd += nx_local;
   }
   if (q < Q-1) num_cols_offd += nx_local*nz_local;
   {
      num_cols_offd += nx_local*nz_local;
      if ( r ) num_cols_offd += nx_local;
      if ( r < R-1 ) num_cols_offd += nx_local;
   }
   if (r) num_cols_offd += nx_local*ny_local;
   if (r < R-1) num_cols_offd += nx_local*ny_local;

   if (!local_num_rows) num_cols_offd = 0;

   if (num_cols_offd) col_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd);

   cnt = 0;
   o_cnt = 0;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = r*nz_local;  iz < (r+1)*nz_local; iz++)
   {
      for (iy = q*ny_local;  iy < (q+1)*ny_local; iy++)
      {
         for (ix = p*nx_local; ix < (p+1)*nx_local; ix++)
         {
            cnt++;
            o_cnt++;
            diag_i[cnt] = diag_i[cnt-1];
            offd_i[o_cnt] = offd_i[o_cnt-1];
            diag_i[cnt]++;
            if (iz > nz_local*r) 
            {
               diag_i[cnt]++;
               if (iy > ny_local*q) 
               {
                  diag_i[cnt]++;
      	          if (ix > nx_local*p)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix) 
      		        offd_i[o_cnt]++;
      	          }
      	          if (ix < nx_local*(p+1)-1)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix+1 < nx) 
      		        offd_i[o_cnt]++;
      	          }
               }
               else
               {
                  if (iy) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx_local*p)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             if (ix < nx_local*(p+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix < nx-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
                  }
               }
               if (ix > nx_local*p) 
                  diag_i[cnt]++;
               else
               {
                  if (ix) 
                  {
                     offd_i[o_cnt]++; 
                  }
               }
               if (ix+1 < nx_local*(p+1)) 
                  diag_i[cnt]++;
               else
               {
                  if (ix+1 < nx) 
                  {
                     offd_i[o_cnt]++; 
                  }
               }
               if (iy+1 < ny_local*(q+1)) 
               {
                  diag_i[cnt]++;
      	          if (ix > nx_local*p)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix) 
      		        offd_i[o_cnt]++;
      	          }
      	          if (ix < nx_local*(p+1)-1)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix+1 < nx) 
      		        offd_i[o_cnt]++;
      	          }
               }
               else
               {
                  if (iy+1 < ny) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx_local*p)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             if (ix < nx_local*(p+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix < nx-1)
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
                  if (iy > ny_local*q) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx_local*p)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix) 
      	   	        offd_i[o_cnt]++;
      	             }
      	             if (ix < nx_local*(p+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix+1 < nx) 
      	   	        offd_i[o_cnt]++;
      	             }
                  }
                  else
                  {
                     if (iy) 
                     {
                        offd_i[o_cnt]++;
      	                if (ix > nx_local*p)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                if (ix < nx_local*(p+1)-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix < nx-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
                     }
                  }
                  if (ix > nx_local*p) 
                     offd_i[o_cnt]++;
                  else
                  {
                     if (ix) 
                     {
                        offd_i[o_cnt]++; 
                     }
                  }
                  if (ix+1 < nx_local*(p+1)) 
                     offd_i[o_cnt]++;
                  else
                  {
                     if (ix+1 < nx) 
                     {
                        offd_i[o_cnt]++; 
                     }
                  }
                  if (iy+1 < ny_local*(q+1)) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx_local*p)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix) 
      	   	           offd_i[o_cnt]++;
      	             }
      	             if (ix < nx_local*(p+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix+1 < nx) 
      	   	           offd_i[o_cnt]++;
      	             }
                  }
                  else
                  {
                     if (iy+1 < ny) 
                     {
                        offd_i[o_cnt]++;
      	                if (ix > nx_local*p)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                if (ix < nx_local*(p+1)-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix < nx-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
                     }
                  }
               }
            }
            if (iy > ny_local*q) 
            {
               diag_i[cnt]++;
   	       if (ix > nx_local*p)
   	       {
   	          diag_i[cnt]++;
   	       }
   	       else
   	       {
   	          if (ix) 
   		     offd_i[o_cnt]++;
   	       }
   	       if (ix < nx_local*(p+1)-1)
   	       {
   	          diag_i[cnt]++;
   	       }
   	       else
   	       {
   	          if (ix+1 < nx) 
   		     offd_i[o_cnt]++;
   	       }
            }
            else
            {
               if (iy) 
               {
                  offd_i[o_cnt]++;
   	          if (ix > nx_local*p)
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          else if (ix)
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          if (ix < nx_local*(p+1)-1)
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          else if (ix < nx-1)
   	          {
   	             offd_i[o_cnt]++;
   	          }
               }
            }
            if (ix > nx_local*p) 
               diag_i[cnt]++;
            else
            {
               if (ix) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (ix+1 < nx_local*(p+1)) 
               diag_i[cnt]++;
            else
            {
               if (ix+1 < nx) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (iy+1 < ny_local*(q+1)) 
            {
               diag_i[cnt]++;
   	       if (ix > nx_local*p)
   	       {
   	          diag_i[cnt]++;
   	       }
   	       else
   	       {
   	          if (ix) 
   		     offd_i[o_cnt]++;
   	       }
   	       if (ix < nx_local*(p+1)-1)
   	       {
   	          diag_i[cnt]++;
   	       }
   	       else
   	       {
   	          if (ix+1 < nx) 
   		     offd_i[o_cnt]++;
   	       }
            }
            else
            {
               if (iy+1 < ny) 
               {
                  offd_i[o_cnt]++;
   	          if (ix > nx_local*p)
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          else if (ix)
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          if (ix < nx_local*(p+1)-1)
   	          {
   	             offd_i[o_cnt]++;
   	          }
   	          else if (ix < nx-1)
   	          {
   	             offd_i[o_cnt]++;
   	          }
               }
            }
            if (iz+1 < nz_local*(r+1)) 
            {
               diag_i[cnt]++;
               if (iy > ny_local*q) 
               {
                  diag_i[cnt]++;
      	          if (ix > nx_local*p)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix) 
      		        offd_i[o_cnt]++;
      	          }
      	          if (ix < nx_local*(p+1)-1)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix+1 < nx) 
      		        offd_i[o_cnt]++;
      	          }
               }
               else
               {
                  if (iy) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx_local*p)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             if (ix < nx_local*(p+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix < nx-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
                  }
               }
               if (ix > nx_local*p) 
                  diag_i[cnt]++;
               else
               {
                  if (ix) 
                  {
                     offd_i[o_cnt]++; 
                  }
               }
               if (ix+1 < nx_local*(p+1)) 
                  diag_i[cnt]++;
               else
               {
                  if (ix+1 < nx) 
                  {
                     offd_i[o_cnt]++; 
                  }
               }
               if (iy+1 < ny_local*(q+1)) 
               {
                  diag_i[cnt]++;
      	          if (ix > nx_local*p)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix) 
      		        offd_i[o_cnt]++;
      	          }
      	          if (ix < nx_local*(p+1)-1)
      	          {
      	             diag_i[cnt]++;
      	          }
      	          else
      	          {
      	             if (ix+1 < nx) 
      		        offd_i[o_cnt]++;
      	          }
               }
               else
               {
                  if (iy+1 < ny) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx_local*p)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             if (ix < nx_local*(p+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else if (ix < nx-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
                  }
               }
            }
            else
            {
               if (iz+1 < nz)
	       {
		  offd_i[o_cnt]++;
                  if (iy > ny_local*q) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx_local*p)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix) 
      	   	        offd_i[o_cnt]++;
      	             }
      	             if (ix < nx_local*(p+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix+1 < nx) 
      	   	        offd_i[o_cnt]++;
      	             }
                  }
                  else
                  {
                     if (iy) 
                     {
                        offd_i[o_cnt]++;
      	                if (ix > nx_local*p)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                if (ix < nx_local*(p+1)-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix < nx-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
                     }
                  }
                  if (ix > nx_local*p) 
                     offd_i[o_cnt]++;
                  else
                  {
                     if (ix) 
                     {
                        offd_i[o_cnt]++; 
                     }
                  }
                  if (ix+1 < nx_local*(p+1)) 
                     offd_i[o_cnt]++;
                  else
                  {
                     if (ix+1 < nx) 
                     {
                        offd_i[o_cnt]++; 
                     }
                  }
                  if (iy+1 < ny_local*(q+1)) 
                  {
                     offd_i[o_cnt]++;
      	             if (ix > nx_local*p)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix) 
      	   	           offd_i[o_cnt]++;
      	             }
      	             if (ix < nx_local*(p+1)-1)
      	             {
      	                offd_i[o_cnt]++;
      	             }
      	             else
      	             {
      	                if (ix+1 < nx) 
      	   	           offd_i[o_cnt]++;
      	             }
                  }
                  else
                  {
                     if (iy+1 < ny) 
                     {
                        offd_i[o_cnt]++;
      	                if (ix > nx_local*p)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                if (ix < nx_local*(p+1)-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
      	                else if (ix < nx-1)
      	                {
      	                   offd_i[o_cnt]++;
      	                }
                     }
                  }
               }
            }
         }
      }
   }

   diag_j = hypre_CTAlloc(HYPRE_Int, diag_i[local_num_rows]);
   diag_data = hypre_CTAlloc(HYPRE_Real, diag_i[local_num_rows]);

   if (num_procs > 1)
   {
      offd_j = hypre_CTAlloc(HYPRE_Int, offd_i[local_num_rows]);
      offd_data = hypre_CTAlloc(HYPRE_Real, offd_i[local_num_rows]);
   }

   nxy = nx_local*ny_local;
   row_index = 0;
   cnt = 0;
   o_cnt = 0;
   for (iz = nz_local*r;  iz < nz_local*(r+1); iz++)
   {
      for (iy = ny_local*q;  iy < ny_local*(q+1); iy++)
      {
         for (ix = nx_local*p; ix < nx_local*(p+1); ix++)
         {
            diag_j[cnt] = row_index;
            diag_data[cnt++] = value[0];
            if (iz > nz_local*r) 
            {
               if (iy > ny_local*q) 
               {
      	          if (ix > nx_local*p)
      	          {
      	             diag_j[cnt] = row_index-nxy-nx_local-1;
      	             diag_data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix) 
      	             {
      		        offd_j[o_cnt] = hypre_map3(ix-1,iy-1,iz-1,p-1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	          }
      	          diag_j[cnt] = row_index-nxy-nx_local;
      	          diag_data[cnt++] = value[1];
      	          if (ix < nx_local*(p+1)-1)
      	          {
      	             diag_j[cnt] = row_index-nxy-nx_local+1;
      	             diag_data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix+1 < nx) 
      	             {
      		        offd_j[o_cnt] = hypre_map3(ix+1,iy-1,iz-1,p+1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	          }
               }
               else
               {
                  if (iy) 
                  {
      	             if (ix > nx_local*p)
      	             {
      		        offd_j[o_cnt] = hypre_map3(ix-1,iy-1,iz-1,p,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else if (ix)
      	             {
      		        offd_j[o_cnt] = hypre_map3(ix-1,iy-1,iz-1,p-1,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      		     offd_j[o_cnt] = hypre_map3(ix,iy-1,iz-1,p,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
      	             if (ix < nx_local*(p+1)-1)
      	             {
      		        offd_j[o_cnt] = hypre_map3(ix+1,iy-1,iz-1,p,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else if (ix < nx-1)
      	             {
      		        offd_j[o_cnt] = hypre_map3(ix+1,iy-1,iz-1,p+1,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
                  }
               }
               if (ix > nx_local*p) 
      	       {   
      	          diag_j[cnt] = row_index-nxy-1;
      	          diag_data[cnt++] = value[1];
      	       }   
               else
               {
                  if (ix) 
                  {
      		     offd_j[o_cnt] = hypre_map3(ix-1,iy,iz-1,p-1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
                  }
               }
      	       diag_j[cnt] = row_index-nxy;
      	       diag_data[cnt++] = value[1];
               if (ix+1 < nx_local*(p+1)) 
      	       {   
      	          diag_j[cnt] = row_index-nxy+1;
      	          diag_data[cnt++] = value[1];
      	       }   
               else
               {
                  if (ix+1 < nx) 
                  {
      		     offd_j[o_cnt] = hypre_map3(ix+1,iy,iz-1,p+1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
                  }
               }
               if (iy+1 < ny_local*(q+1)) 
               {
      	          if (ix > nx_local*p)
      	          {
      	             diag_j[cnt] = row_index-nxy+nx_local-1;
      	             diag_data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix) 
                     {
      		        offd_j[o_cnt] = hypre_map3(ix-1,iy+1,iz-1,p-1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
                     }
      	          }
      	          diag_j[cnt] = row_index-nxy+nx_local;
      	          diag_data[cnt++] = value[1];
      	          if (ix < nx_local*(p+1)-1)
      	          {
      	             diag_j[cnt] = row_index-nxy+nx_local+1;
      	             diag_data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix+1 < nx) 
                     {
      		        offd_j[o_cnt] = hypre_map3(ix+1,iy+1,iz-1,p+1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
                     }
      	          }
               }
               else
               {
                  if (iy+1 < ny) 
                  {
      	             if (ix > nx_local*p)
      	             {
      		        offd_j[o_cnt] = hypre_map3(ix-1,iy+1,iz-1,p,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else if (ix)
      	             {
      		        offd_j[o_cnt] = hypre_map3(ix-1,iy+1,iz-1,p-1,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      		     offd_j[o_cnt] = hypre_map3(ix,iy+1,iz-1,p,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
      	             if (ix < nx_local*(p+1)-1)
      	             {
      		        offd_j[o_cnt] = hypre_map3(ix+1,iy+1,iz-1,p,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else if (ix < nx-1)
      	             {
      		        offd_j[o_cnt] = hypre_map3(ix+1,iy+1,iz-1,p+1,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
                  }
               }
            }
            else
            {
               if (iz)
	       {
                  if (iy > ny_local*q) 
                  {
      	             if (ix > nx_local*p)
      	             {
      		        offd_j[o_cnt] = hypre_map3(ix-1,iy-1,iz-1,p,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix) 
      	                {
      		           offd_j[o_cnt] = hypre_map3(ix-1,iy-1,iz-1,p-1,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	             }
      		     offd_j[o_cnt] = hypre_map3(ix,iy-1,iz-1,p,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
      	             if (ix < nx_local*(p+1)-1)
      	             {
      		        offd_j[o_cnt] = hypre_map3(ix+1,iy-1,iz-1,p,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix+1 < nx) 
      	                {
      		           offd_j[o_cnt] = hypre_map3(ix+1,iy-1,iz-1,p+1,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	             }
                  }
                  else
                  {
                     if (iy) 
                     {
      	                if (ix > nx_local*p)
      	                {
      		           offd_j[o_cnt] = hypre_map3(ix-1,iy-1,iz-1,p,q-1,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	                else if (ix)
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix-1,iy-1,iz-1,p-1,q-1,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      		        offd_j[o_cnt] = hypre_map3(ix,iy-1,iz-1,p,q-1,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	                if (ix < nx_local*(p+1)-1)
      	                {
      		           offd_j[o_cnt] = hypre_map3(ix+1,iy-1,iz-1,p,q-1,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	                else if (ix < nx-1)
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix+1,iy-1,iz-1,p+1,q-1,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
                     }
                  }
                  if (ix > nx_local*p) 
                  {
      		     offd_j[o_cnt] =hypre_map3(ix-1,iy,iz-1,p,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
                  }
                  else
                  {
                     if (ix) 
                     {
      		        offd_j[o_cnt] =hypre_map3(ix-1,iy,iz-1,p-1,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
                     }
                  }
      		  offd_j[o_cnt] =hypre_map3(ix,iy,iz-1,p,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		  offd_data[o_cnt++] = value[1];
                  if (ix+1 < nx_local*(p+1)) 
                  {
      		     offd_j[o_cnt] =hypre_map3(ix+1,iy,iz-1,p,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
                  }
                  else
                  {
                     if (ix+1 < nx) 
                     {
      		        offd_j[o_cnt] =hypre_map3(ix+1,iy,iz-1,p+1,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
                     }
                  }
                  if (iy+1 < ny_local*(q+1)) 
                  {
      	             if (ix > nx_local*p)
      	             {
      		        offd_j[o_cnt] =hypre_map3(ix-1,iy+1,iz-1,p,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix) 
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix-1,iy+1,iz-1,p-1,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	             }
      		     offd_j[o_cnt] =hypre_map3(ix,iy+1,iz-1,p,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
      	             if (ix < nx_local*(p+1)-1)
      	             {
      		        offd_j[o_cnt] =hypre_map3(ix+1,iy+1,iz-1,p,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix+1 < nx) 
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix+1,iy+1,iz-1,p+1,q,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	             }
                  }
                  else
                  {
                     if (iy+1 < ny) 
                     {
      	                if (ix > nx_local*p)
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix-1,iy+1,iz-1,p,q+1,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	                else if (ix)
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix-1,iy+1,iz-1,p-1,q+1,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      		        offd_j[o_cnt] =hypre_map3(ix,iy+1,iz-1,p,q+1,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	                if (ix < nx_local*(p+1)-1)
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix+1,iy+1,iz-1,p,q+1,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	                else if (ix < nx-1)
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix+1,iy+1,iz-1,p+1,q+1,r-1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
                     }
                  }
               }
            }
            if (iy > ny_local*q) 
            {
   	       if (ix > nx_local*p)
   	       {
   	          diag_j[cnt] = row_index-nx_local-1;
   	          diag_data[cnt++] = value[1];
   	       }
   	       else
   	       {
   	          if (ix) 
   	          {
      		     offd_j[o_cnt] =hypre_map3(ix-1,iy-1,iz,p-1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
   	          }
   	       }
   	       diag_j[cnt] = row_index-nx_local;
   	       diag_data[cnt++] = value[1];
   	       if (ix < nx_local*(p+1)-1)
   	       {
   	          diag_j[cnt] = row_index-nx_local+1;
   	          diag_data[cnt++] = value[1];
   	       }
   	       else
   	       {
   	          if (ix+1 < nx) 
   	          {
      		     offd_j[o_cnt] =hypre_map3(ix+1,iy-1,iz,p+1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
   	          }
   	       }
            }
            else
            {
               if (iy) 
               {
   	          if (ix > nx_local*p)
   	          {
      		     offd_j[o_cnt] =hypre_map3(ix-1,iy-1,iz,p,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
   	          }
   	          else if (ix)
   	          {
      		     offd_j[o_cnt] =hypre_map3(ix-1,iy-1,iz,p-1,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
   	          }
      		  offd_j[o_cnt] =hypre_map3(ix,iy-1,iz,p,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		  offd_data[o_cnt++] = value[1];
   	          if (ix < nx_local*(p+1)-1)
   	          {
      		     offd_j[o_cnt] =hypre_map3(ix+1,iy-1,iz,p,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
   	          }
   	          else if (ix < nx-1)
   	          {
      		     offd_j[o_cnt] =hypre_map3(ix+1,iy-1,iz,p+1,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
   	          }
               }
            }
            if (ix > nx_local*p) 
            {
               diag_j[cnt] = row_index-1;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix) 
               {
      		  offd_j[o_cnt] =hypre_map3(ix-1,iy,iz,p-1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		  offd_data[o_cnt++] = value[1];
               }
            }
            if (ix+1 < nx_local*(p+1)) 
            {
               diag_j[cnt] = row_index+1;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ix+1 < nx) 
               {
      		  offd_j[o_cnt] =hypre_map3(ix+1,iy,iz,p+1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		  offd_data[o_cnt++] = value[1];
               }
            }
            if (iy+1 < ny_local*(q+1)) 
            {
   	       if (ix > nx_local*p)
   	       {
                  diag_j[cnt] = row_index+nx_local-1;
                  diag_data[cnt++] = value[1];
   	       }
   	       else
   	       {
   	          if (ix) 
                  {
      		     offd_j[o_cnt] =hypre_map3(ix-1,iy+1,iz,p-1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
                  }
   	       }
               diag_j[cnt] = row_index+nx_local;
               diag_data[cnt++] = value[1];
   	       if (ix < nx_local*(p+1)-1)
   	       {
                  diag_j[cnt] = row_index+nx_local+1;
                  diag_data[cnt++] = value[1];
   	       }
   	       else
   	       {
   	          if (ix+1 < nx) 
                  {
      		     offd_j[o_cnt] =hypre_map3(ix+1,iy+1,iz,p+1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
                  }
   	       }
            }
            else
            {
               if (iy+1 < ny) 
               {
   	          if (ix > nx_local*p)
   	          {
      		     offd_j[o_cnt] =hypre_map3(ix-1,iy+1,iz,p,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
   	          }
   	          else if (ix)
   	          {
      		     offd_j[o_cnt] =hypre_map3(ix-1,iy+1,iz,p-1,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
   	          }
      		  offd_j[o_cnt] =hypre_map3(ix,iy+1,iz,p,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		  offd_data[o_cnt++] = value[1];
   	          if (ix < nx_local*(p+1)-1)
   	          {
      		     offd_j[o_cnt] =hypre_map3(ix+1,iy+1,iz,p,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
   	          }
   	          else if (ix < nx-1)
   	          {
      		     offd_j[o_cnt] =hypre_map3(ix+1,iy+1,iz,p+1,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
   	          }
               }
            }
            if (iz+1 < nz_local*(r+1)) 
            {
               if (iy > ny_local*q) 
               {
      	          if (ix > nx_local*p)
      	          {
      	             diag_j[cnt] = row_index+nxy-nx_local-1;
      	             diag_data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix) 
   	             {
      		        offd_j[o_cnt] =hypre_map3(ix-1,iy-1,iz+1,p-1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
   	             }
      	          }
      	          diag_j[cnt] = row_index+nxy-nx_local;
      	          diag_data[cnt++] = value[1];
      	          if (ix < nx_local*(p+1)-1)
      	          {
      	             diag_j[cnt] = row_index+nxy-nx_local+1;
      	             diag_data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix+1 < nx) 
   	             {
      		        offd_j[o_cnt] =hypre_map3(ix+1,iy-1,iz+1,p+1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
   	             }
      	          }
               }
               else
               {
                  if (iy) 
                  {
      	             if (ix > nx_local*p)
      	             {
      		        offd_j[o_cnt] =hypre_map3(ix-1,iy-1,iz+1,p,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else if (ix)
      	             {
      		        offd_j[o_cnt] =hypre_map3(ix-1,iy-1,iz+1,p-1,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      		     offd_j[o_cnt] =hypre_map3(ix,iy-1,iz+1,p,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
      	             if (ix < nx_local*(p+1)-1)
      	             {
      		        offd_j[o_cnt] =hypre_map3(ix+1,iy-1,iz+1,p,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else if (ix < nx-1)
      	             {
      		        offd_j[o_cnt] =hypre_map3(ix+1,iy-1,iz+1,p+1,q-1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
                  }
               }
               if (ix > nx_local*p) 
               {
                  diag_j[cnt] = row_index+nxy-1;
                  diag_data[cnt++] = value[1];
               }
               else
               {
                  if (ix) 
                  {
      		     offd_j[o_cnt] =hypre_map3(ix-1,iy,iz+1,p-1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
                  }
               }
               diag_j[cnt] = row_index+nxy;
               diag_data[cnt++] = value[1];
               if (ix+1 < nx_local*(p+1)) 
               {
                  diag_j[cnt] = row_index+nxy+1;
                  diag_data[cnt++] = value[1];
               }
               else
               {
                  if (ix+1 < nx) 
                  {
      		     offd_j[o_cnt] =hypre_map3(ix+1,iy,iz+1,p+1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
                  }
               }
               if (iy+1 < ny_local*(q+1)) 
               {
      	          if (ix > nx_local*p)
      	          {
                     diag_j[cnt] = row_index+nxy+nx_local-1;
                     diag_data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix) 
                     {
      		        offd_j[o_cnt] =hypre_map3(ix-1,iy+1,iz+1,p-1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
                     }
      	          }
                  diag_j[cnt] = row_index+nxy+nx_local;
                  diag_data[cnt++] = value[1];
      	          if (ix < nx_local*(p+1)-1)
      	          {
                     diag_j[cnt] = row_index+nxy+nx_local+1;
                     diag_data[cnt++] = value[1];
      	          }
      	          else
      	          {
      	             if (ix+1 < nx) 
                     {
      		        offd_j[o_cnt] =hypre_map3(ix+1,iy+1,iz+1,p+1,q,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
                     }
      	          }
               }
               else
               {
                  if (iy+1 < ny) 
                  {
      	             if (ix > nx_local*p)
      	             {
      		        offd_j[o_cnt] =hypre_map3(ix-1,iy+1,iz+1,p,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else if (ix)
      	             {
      		        offd_j[o_cnt] =hypre_map3(ix-1,iy+1,iz+1,p-1,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      		     offd_j[o_cnt] =hypre_map3(ix,iy+1,iz+1,p,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
      	             if (ix < nx_local*(p+1)-1)
      	             {
      		        offd_j[o_cnt] =hypre_map3(ix+1,iy+1,iz+1,p,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else if (ix < nx-1)
      	             {
      		        offd_j[o_cnt] =hypre_map3(ix+1,iy+1,iz+1,p+1,q+1,r,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
                  }
               }
            }
            else
            {
               if (iz+1 < nz)
	       {
                  if (iy > ny_local*q) 
                  {
      	             if (ix > nx_local*p)
      	             {
      		        offd_j[o_cnt] =hypre_map3(ix-1,iy-1,iz+1,p,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix) 
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix-1,iy-1,iz+1,p-1,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	             }
      		     offd_j[o_cnt] =hypre_map3(ix,iy-1,iz+1,p,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
      	             if (ix < nx_local*(p+1)-1)
      	             {
      		        offd_j[o_cnt] =hypre_map3(ix+1,iy-1,iz+1,p,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix+1 < nx) 
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix+1,iy-1,iz+1,p+1,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	             }
                  }
                  else
                  {
                     if (iy) 
                     {
      	                if (ix > nx_local*p)
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix-1,iy-1,iz+1,p,q-1,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	                else if (ix)
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix-1,iy-1,iz+1,p-1,q-1,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      		        offd_j[o_cnt] =hypre_map3(ix,iy-1,iz+1,p,q-1,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	                if (ix < nx_local*(p+1)-1)
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix+1,iy-1,iz+1,p,q-1,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	                else if (ix < nx-1)
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix+1,iy-1,iz+1,p+1,q-1,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
                     }
                  }
                  if (ix > nx_local*p) 
                  {
      		     offd_j[o_cnt] =hypre_map3(ix-1,iy,iz+1,p,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
                  }
                  else
                  {
                     if (ix) 
                     {
      		        offd_j[o_cnt] =hypre_map3(ix-1,iy,iz+1,p-1,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
                     }
                  }
      		  offd_j[o_cnt] =hypre_map3(ix,iy,iz+1,p,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		  offd_data[o_cnt++] = value[1];
                  if (ix+1 < nx_local*(p+1)) 
                  {
      		     offd_j[o_cnt] =hypre_map3(ix+1,iy,iz+1,p,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
                  }
                  else
                  {
                     if (ix+1 < nx) 
                     {
      		        offd_j[o_cnt] =hypre_map3(ix+1,iy,iz+1,p+1,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
                     }
                  }
                  if (iy+1 < ny_local*(q+1)) 
                  {
      	             if (ix > nx_local*p)
      	             {
      		        offd_j[o_cnt] =hypre_map3(ix-1,iy+1,iz+1,p,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix) 
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix-1,iy+1,iz+1,p-1,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	             }
      		     offd_j[o_cnt] =hypre_map3(ix,iy+1,iz+1,p,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		     offd_data[o_cnt++] = value[1];
      	             if (ix < nx_local*(p+1)-1)
      	             {
      		        offd_j[o_cnt] =hypre_map3(ix+1,iy+1,iz+1,p,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	             }
      	             else
      	             {
      	                if (ix+1 < nx) 
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix+1,iy+1,iz+1,p+1,q,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	             }
                  }
                  else
                  {
                     if (iy+1 < ny) 
                     {
      	                if (ix > nx_local*p)
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix-1,iy+1,iz+1,p,q+1,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	                else if (ix)
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix-1,iy+1,iz+1,p-1,q+1,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      		        offd_j[o_cnt] =hypre_map3(ix,iy+1,iz+1,p,q+1,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		        offd_data[o_cnt++] = value[1];
      	                if (ix < nx_local*(p+1)-1)
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix+1,iy+1,iz+1,p,q+1,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
      	                else if (ix < nx-1)
      	                {
      		           offd_j[o_cnt] =hypre_map3(ix+1,iy+1,iz+1,p+1,q+1,r+1,P,Q,R,
					nx_part,ny_part,nz_part,global_part);
      		           offd_data[o_cnt++] = value[1];
      	                }
                     }
                  }
               }
            }
            row_index++;
         }
      }
   }

   if (num_procs > 1)
   {
      work = hypre_CTAlloc(HYPRE_Int,o_cnt);

      for (i=0; i < o_cnt; i++)
         work[i] = offd_j[i];

      hypre_qsort0(work, 0, o_cnt-1);

      col_map_offd[0] = work[0];
      cnt = 0;
      for (i=0; i < o_cnt; i++)
      {
         if (work[i] > col_map_offd[cnt])
         {
            cnt++;
            col_map_offd[cnt] = work[i];
         }
      }

      for (i=0; i < o_cnt; i++)
      {
         for (j=0; j < num_cols_offd; j++)
         {
            if (offd_j[i] == col_map_offd[j])
            {
               offd_j[i] = j;
               break;
            }
         }
      }

      hypre_TFree(work);
   }



#ifdef HYPRE_NO_GLOBAL_PARTITION
   global_part = hypre_CTAlloc(HYPRE_Int, 2);
   global_part[0] = my_id*local_num_rows;
   global_part[1] = (my_id+1)*local_num_rows;
#else
   global_part = hypre_CTAlloc(HYPRE_Int, num_procs);
   for (i=0; i < num_procs+1; i++)
      global_part[i] = i*local_num_rows;
#endif


   A = hypre_ParCSRMatrixCreate(comm, grid_size, grid_size,
                                global_part, global_part, num_cols_offd,
                                diag_i[local_num_rows],
                                offd_i[local_num_rows]);

   hypre_ParCSRMatrixColMapOffd(A) = col_map_offd;

   diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_data;

   offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrixI(offd) = offd_i;
   if (num_cols_offd)
   {
      hypre_CSRMatrixJ(offd) = offd_j;
      hypre_CSRMatrixData(offd) = offd_data;
   }


   return (HYPRE_ParCSRMatrix) A;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_map3( HYPRE_Int  ix,
      HYPRE_Int  iy,
      HYPRE_Int  iz,
      HYPRE_Int  p,
      HYPRE_Int  q,
      HYPRE_Int  r,
      HYPRE_Int  P,
      HYPRE_Int  Q,
      HYPRE_Int  R,
      HYPRE_Int  nx_local,
      HYPRE_Int  ny_local,
      HYPRE_Int  nz_local)
{
   HYPRE_Int ix_local;
   HYPRE_Int iy_local;
   HYPRE_Int iz_local;
   HYPRE_Int nxy;
   HYPRE_Int global_index;
   HYPRE_Int proc_num;
 
   proc_num = r*P*Q + q*P + p;
   nxy = nx_local*ny_local;
   ix_local = ix - nx_local*p;
   iy_local = iy - ny_local*q;
   iz_local = iz - nz_local*r;
   global_index = proc_num*nx_local*ny_local*nz_local 
      + iz_local*nxy + iy_local*nx_local + ix_local;

   return global_index;
}
