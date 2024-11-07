# Code to retrieve calls for all cells at a given position
# from a metdense2 dataset

library( mmap )
library( inline )
library(tidyverse)

metdense_path <- "/home/anders//w/metdense2/scnmt_data__CpG__filt.metdense"
chr1 <- list(
  pos = mmap( file.path( metdense_path, "1.pos" ), int32(), prot=mmapFlags(PROT_READ) ),
  dat = mmap( file.path( metdense_path, "1.dat" ), int32(), prot=mmapFlags(PROT_READ) ) )


meta <- jsonlite::read_json( file.path( metdense_path, "metadata.json" ), simplifyVector=TRUE )
ncells <- length( meta$cells )

pos_idx <- 1
storage.mode( chr1$dat[ ( (pos_idx-1) * ceiling( ncells / 16 ) + 1 ) : ( pos_idx * ceiling( ncells / 16 ) ) ] )

fun <- cfunction(
  signature( dat="integer", ncells="integer" ),
  body="
   int n = INTEGER(ncells)[0];
   
   // count cells with calls
   int k = 0;
   for( int i = 0; i < LENGTH(dat); i++ ) {
      uint32_t word = (uint32_t) INTEGER(dat)[i];
      for( int j = 0; j < 16; j++ ) {
         if( i*16 + j > n )
            break;
         if( word & 3 ) 
            k++;
         word >>= 2;
      }
   }
   
   // allocate return objects
   SEXP list, cell, call, names;

   PROTECT(list = allocVector(VECSXP, 2));
   PROTECT(cell = allocVector(INTSXP, k));
   PROTECT(call = allocVector(INTSXP, k));

   SET_VECTOR_ELT(list, 0, cell);
   SET_VECTOR_ELT(list, 1, call);

   PROTECT(names = allocVector(STRSXP, 2));
   SET_STRING_ELT(names, 0, mkChar(\"cell\"));
   SET_STRING_ELT(names, 1, mkChar(\"call\"));

   setAttrib(list, R_NamesSymbol, names);

   // put values
   int ki = 0;
   for( int i = 0; i < LENGTH(dat); i++ ) {
      uint32_t word = (uint32_t) INTEGER(dat)[i];
      for( int j = 0; j < 16; j++ ) {
         if( i*16 + j > n )
            break;
         if( word & 3 ) {
            INTEGER(cell)[ki] = i*16 + j + 1;
            INTEGER(call)[ki] = word & 3;
            ki++;
         }
         word >>= 2;
      }
   }


   UNPROTECT(4);   
   return list;
   ",
  includes = "#include <stdint.h>"
)

pos_idx <- 198
res <- fun( chr1$dat[ ( (pos_idx-1) * ceiling( ncells / 16 ) + 1 ) : ( pos_idx * ceiling( ncells / 16 ) ) ], ncells )
as_tibble(res)
