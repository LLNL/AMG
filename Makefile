#BHEADER**********************************************************************
# Copyright (c) 2017,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Ulrike Yang (yang11@llnl.gov) et al. CODE-LLNL-738-322.
# This file is part of AMG.  See files COPYRIGHT and README for details.
#
# AMG is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
#EHEADER**********************************************************************

srcdir = .

HYPRE_DIRS =\
  utilities\
  krylov\
  IJ_mv\
  parcsr_ls\
  parcsr_mv\
  seq_mv

all: $(HYPRE_DIRS)
	$(MAKE) -C test
$(HYPRE_DIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean: $(HYPRE_DIRS)
	$(MAKE) -C test clean

veryclean: $(HYPRE_DIRS)
	$(MAKE) -C test veryclean

tags:
	find . -name "*.c" -or -name "*.C" -or -name "*.h" -or -name "*.c??" -or -name "*.h??" -or -name "*.f" | etags -

.PHONY: all $(HYPRE_DIRS)
