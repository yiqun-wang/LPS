//
// MATLAB Compiler: 6.2 (R2016a)
// Date: Mon Nov 12 17:47:20 2018
// Arguments: "-B" "macro_default" "-W" "cpplib:libcompcur" "-T" "link:lib"
// "compute_curvature.m" 
//

#ifndef __libcompcur_h
#define __libcompcur_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#include "mclcppclass.h"
#ifdef __cplusplus
extern "C" {
#endif

#if defined(__SUNPRO_CC)
/* Solaris shared libraries use __global, rather than mapfiles
 * to define the API exported from a shared library. __global is
 * only necessary when building the library -- files including
 * this header file to use the library do not need the __global
 * declaration; hence the EXPORTING_<library> logic.
 */

#ifdef EXPORTING_libcompcur
#define PUBLIC_libcompcur_C_API __global
#else
#define PUBLIC_libcompcur_C_API /* No import statement needed. */
#endif

#define LIB_libcompcur_C_API PUBLIC_libcompcur_C_API

#elif defined(_HPUX_SOURCE)

#ifdef EXPORTING_libcompcur
#define PUBLIC_libcompcur_C_API __declspec(dllexport)
#else
#define PUBLIC_libcompcur_C_API __declspec(dllimport)
#endif

#define LIB_libcompcur_C_API PUBLIC_libcompcur_C_API


#else

#define LIB_libcompcur_C_API

#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_libcompcur_C_API 
#define LIB_libcompcur_C_API /* No special import/export declaration */
#endif

extern LIB_libcompcur_C_API 
bool MW_CALL_CONV libcompcurInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_libcompcur_C_API 
bool MW_CALL_CONV libcompcurInitialize(void);

extern LIB_libcompcur_C_API 
void MW_CALL_CONV libcompcurTerminate(void);



extern LIB_libcompcur_C_API 
void MW_CALL_CONV libcompcurPrintStackTrace(void);

extern LIB_libcompcur_C_API 
bool MW_CALL_CONV mlxCompute_curvature(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                       *prhs[]);


#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

/* On Windows, use __declspec to control the exported API */
#if defined(_MSC_VER) || defined(__BORLANDC__)

#ifdef EXPORTING_libcompcur
#define PUBLIC_libcompcur_CPP_API __declspec(dllexport)
#else
#define PUBLIC_libcompcur_CPP_API __declspec(dllimport)
#endif

#define LIB_libcompcur_CPP_API PUBLIC_libcompcur_CPP_API

#else

#if !defined(LIB_libcompcur_CPP_API)
#if defined(LIB_libcompcur_C_API)
#define LIB_libcompcur_CPP_API LIB_libcompcur_C_API
#else
#define LIB_libcompcur_CPP_API /* empty! */ 
#endif
#endif

#endif

extern LIB_libcompcur_CPP_API void MW_CALL_CONV compute_curvature(int nargout, mwArray& Umax, mwArray& Umin, mwArray& Cmax, mwArray& Cmin, mwArray& Normal, mwArray& hks, mwArray& diameter, mwArray& resp, mwArray& le, mwArray& cf, const mwArray& V, const mwArray& F, const mwArray& off_filename, const mwArray& hks_len, const mwArray& le_len, const mwArray& cf_flag);

#endif
#endif
