#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************

.SUFFIXES:
.SUFFIXES: .c .f .o

include ../Makefile.include

CFLAGS =\
 -I..\
 -I../utilities\
 -I../krylov\
 -I../seq_mv\
 -I../parcsr_mv\
 -I../IJ_mv\
 ${INCLUDE_CFLAGS}

HEADERS =\
 ams.h\
 HYPRE_parcsr_ls.h\
 _hypre_parcsr_ls.h\
 aux_interp.h\
 headers.h\
 par_amg.h

FILES =\
 ams.c\
 aux_interp.c\
 gen_redcs_mat.c\
 HYPRE_parcsr_amg.c\
 HYPRE_parcsr_gmres.c\
 HYPRE_parcsr_pcg.c\
 par_amg.c\
 par_amg_setup.c\
 par_amg_solve.c\
 par_cg_relax_wt.c\
 par_coarsen.c\
 par_cheby.c\
 par_coarse_parms.c\
 par_cycle.c\
 par_add_cycle.c\
 par_difconv.c\
 par_indepset.c\
 par_interp.c\
 par_multi_interp.c\
 par_lr_interp.c\
 par_nongalerkin.c\
 par_rap.c\
 par_rap_communication.c\
 par_relax.c\
 par_relax_more.c\
 par_relax_interface.c\
 par_scaled_matnorm.c\
 par_stats.c\
 par_strength.c\
 partial.c\
 pcg_par.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libparcsr_ls.a

clean:
	@rm -f *.o *.dxx

veryclean: clean
	@rm -f lib*.a

##################################################################
# Rules
##################################################################

libparcsr_ls.a: ${OBJS}
	@echo  "Building $@ ... "
	ar -rcu $@ ${OBJS}
	ranlib $@

${OBJS}: ${HEADERS}

##################################################################
# Generic rules
##################################################################

.c.o:
	${CC} -o $@ -c ${CFLAGS} $<
