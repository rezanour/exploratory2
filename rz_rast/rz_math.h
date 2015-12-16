#pragma once

// Basic FPU based math library used in rz_rast.
// Types and operators are only being added as needed,
// which is why it may look incomplete.

#include <math.h>
#include <float.h>
#include <assert.h>

struct alignas(8) float2
{
    float x, y;

    // constructors
    float2()    // no default init for perf
    {}

    float2(float x, float y)
        : x(x), y(y)
    {}

    float2(int x, int y)
        : x((float)x), y((float)y)
    {}

    float2(const float2& other)
        : x(other.x), y(other.y)
    {}

    // assignment
    float2& operator= (const float2& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }
};

struct alignas(16) float3
{
    float x, y, z;

    // constructors
    float3()    // no default init for perf
    {}

    float3(float x, float y, float z)
        : x(x), y(y), z(z)
    {}

    float3(const float3& other)
        : x(other.x), y(other.y), z(other.z)
    {}

    // assignment
    float3& operator= (const float3& other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }

    // operations
    float lengthSq() const { return x * x + y * y + z * z; }
    float length() const { return sqrtf(lengthSq()); }

private:
    float pad;
};

struct alignas(16) float4
{
    float x, y, z, w;

    // constructors
    float4()    // no default init for perf
    {}

    float4(float x, float y, float z, float w)
        : x(x), y(y), z(z), w(w)
    {}

    float4(const float4& other)
        : x(other.x), y(other.y), z(other.z), w(other.w)
    {}

    float4(const float3& other, float w)
        : x(other.x), y(other.y), z(other.z), w(w)
    {}

    // assignment
    float4& operator= (const float4& other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        w = other.w;
        return *this;
    }

    float4& operator /= (float s)
    {
        x /= s;
        y /= s;
        z /= s;
        w /= s;
        return *this;
    }
};

#pragma warning(push)
#pragma warning(disable: 4201)  // nameless struct/union

struct alignas(16) matrix4x4
{
    // contents can be accessed in any of the following ways:
    //     flat list of 16 floats,
    //     2d array of floats
    //     rows of 4 float4
    union
    {
        float f[16];

        float m[4][4];

        struct alignas(16)
        {
            float4 r[4];
        };
    };

    // constructors
    matrix4x4()    // no default init for perf
    {}

    matrix4x4(
        float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33)
        : f { m00, m01, m02, m03, m10, m11, m12, m13,
        m20, m21, m22, m23, m30, m31, m32, m33 }
    {
    }

    matrix4x4(const matrix4x4& other)
        : r { other.r[0], other.r[1], other.r[2], other.r[3] }
    {
    }
};

#pragma warning(pop)

// operators
float2 operator- (const float2& v1, const float2& v2)
{
    return float2(v1.x - v2.x, v1.y - v2.y);
}

float3 operator+ (const float3& v1, const float3& v2)
{
    return float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

float3 operator- (const float3& v1, const float3& v2)
{
    return float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

float3 operator* (const float3& v, float s)
{
    return float3(v.x * s, v.y * s, v.z * s);
}

float4 operator+ (const float4& v1, const float4& v2)
{
    return float4(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v2.w + v2.w);
}

float4 operator* (const float4& v, float s)
{
    return float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

inline float dot(const float2& v1, const float2& v2)
{
    return v1.x * v2.x + v1.y * v2.y;
}

inline float dot(const float3& v1, const float3& v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline float dot(const float4& v1, const float4& v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

inline float3 cross(const float3& v1, const float3& v2)
{
    return float3(
        v1.y * v2.z - v2.y * v1.z,
        v1.z * v2.x - v2.z * v1.x,
        v1.x * v2.y - v2.x * v1.y);
}

inline float4 mul(const matrix4x4& m, const float4& v)
{
    return float4(
        dot(float4(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0]), v),
        dot(float4(m.m[0][1], m.m[1][1], m.m[2][1], m.m[3][1]), v),
        dot(float4(m.m[0][2], m.m[1][2], m.m[2][2], m.m[3][2]), v),
        dot(float4(m.m[0][3], m.m[1][3], m.m[2][3], m.m[3][3]), v));
}

inline float min(float a, float b)
{
    return a < b ? a : b;
}

inline float max(float a, float b)
{
    return a > b ? a : b;
}

inline float2 min(const float2& v1, const float2& v2)
{
    return float2(min(v1.x, v2.x), min(v1.y, v2.y));
}

inline float2 max(const float2& v1, const float2& v2)
{
    return float2(max(v1.x, v2.x), max(v1.y, v2.y));
}

inline float3 min(const float3& v1, const float3& v2)
{
    return float3(min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z));
}

inline float3 max(const float3& v1, const float3& v2)
{
    return float3(max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z));
}

