#pragma once

#include "vector_types.h"
#include <string>
#include <assert.h>

typedef uchar4 rgb;

typedef float4 vertex; // x, h, z

typedef float4 gridpoint; // h, uh, vh, b

typedef int reflection;

#define M_PI 3.14159265358979323846
#define M_E 2.71828182845904523536
#define EPSILON 0.000001

#define cudaCheck(x) { cudaError_t err = x;  }
#define synchronCheck {}
