/******************************************************************************
* lib3d.h
******************************************************************************/
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error enumeration
typedef enum l3d_error
{
    l3d_no_error = 0,
    l3d_invalid_param,
} l3d_error;

l3d_error l3d_initialize();

#ifdef __cplusplus
} // extern  "C"
#endif
