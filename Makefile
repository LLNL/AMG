#BHEADER**********************************************************************
#  Copyright (c) 2014,  Lawrence Livermore National Security, LLC.
#  Produced at the Lawrence Livermore National Laboratory.
#  Written by Rob Falgout and Ulrike Yang (umyang@llnl.gov).
#  LLNL-CODE-659229. All rights reserved.
# 
#  This file is part of AMG2013. For details, see the docs directory. 
# 
#  Please also read files COPYRIGHT and COPYING_LESSER for details.
# 
#  AMG2013 is free software; you can redistribute it and/or modify it under the
#  terms of the GNU Lesser General Public License (as published by the Free
#  Software Foundation) version 2.1 dated February 1999.
# 
#  This program is distributed in the hope that it will be useful, but WITHOUT 
#  ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or 
#  FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the GNU 
#  General Public License for more details.
#  
#  You should have received a copy of the GNU Lesser General Public License 
#  along with this program; if not, write to the Free Software Foundation, 
#  Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#EHEADER**********************************************************************



srcdir = .

HYPRE_DIRS =\
 utilities\
 krylov\
 IJ_mv\
 parcsr_ls\
 parcsr_mv\
 seq_mv\
 test

all:
	@ \
	for i in ${HYPRE_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Making $$i ..."; \
	    (cd $$i; make); \
	    echo ""; \
	  fi; \
	done

clean:
	@ \
	for i in ${HYPRE_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Cleaning $$i ..."; \
	    (cd $$i; make clean); \
	  fi; \
	done

veryclean:
	@ \
	for i in ${HYPRE_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Very-cleaning $$i ..."; \
	    (cd $$i; make veryclean); \
	  fi; \
	done

tags:
	find . -name "*.c" -or -name "*.C" -or -name "*.h" -or -name "*.c??" -or -name "*.h??" -or -name "*.f" | etags -
