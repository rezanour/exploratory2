//==============================================================================
// Simple GPU ComputeShader based voxel ray tracer.
// Reza Nourai
//==============================================================================

// Used to compute primary eye rays
cbuffer CameraData
{
    float4x4 CameraWorldTransform;
    float2 HalfSize;
    float DistToProjPlane;
};

struct AABB
{
    float3 Center;
    float3 HalfWidths;
};

// Scene objects
StructuredBuffer<AABB> Blocks;

// Output RenderTarget (not marked globally coherent because we're only writing
// from the thread groups, and also writing to unique locations)
RWTexture2D<float4> RenderTarget;

//==============================================================================
// Ray/Block Test - Simple test to see if ray hits the block
//==============================================================================

bool RayBlockTest(float3 start, float3 dir, AABB block)
{
    float3 nonAbsDist = start - block.Center;
    float3 totalDist = abs(nonAbsDist);
    float3 dist = totalDist - block.HalfWidths;
    float3 startDirCheck = nonAbsDist * dir;
    float3 scaleValue = abs(dist * rcp(dir));

    // x
    if (startDirCheck.x < 0 && dist.x > 0)
    {
        float3 p = start + dir * scaleValue.x;
        float3 test = abs(p - block.Center);
        if (test.y <= block.HalfWidths.y && test.z <= block.HalfWidths.z)
        {
            // hit
            return true;
        }
    }
    // y
    if (startDirCheck.y < 0 && dist.y > 0)
    {
        float3 p = start + dir * scaleValue.y;
        float3 test = abs(p - block.Center);
        if (test.x <= block.HalfWidths.x && test.z <= block.HalfWidths.z)
        {
            // hit
            return true;
        }
    }
    // z
    if (startDirCheck.z < 0 && dist.z > 0)
    {
        float3 p = start + dir * scaleValue.z;
        float3 test = abs(p - block.Center);
        if (test.x <= block.HalfWidths.x && test.y <= block.HalfWidths.y)
        {
            // hit
            return true;
        }
    }

    return false;
}

//==============================================================================
// RayTrace - See if any hits in the scene
//==============================================================================

bool RayTrace(float3 start, float3 dir)
{
    uint numBlocks;
    uint stride;
    Blocks.GetDimensions(numBlocks, stride);

    for (uint i = 0; i < numBlocks; ++i)
    {
        if (RayBlockTest(start, dir, Blocks[i]))
        {
            return true;
        }
    }

    return false;
}

//==============================================================================
// Main entry point
//==============================================================================

// 4x4 pixel blocks
[numthreads(4, 4, 1)]
void main(uint3 GroupThreadID : SV_GroupThreadID, uint3 GroupID : SV_GroupID)
{
    // Determine which pixel we correspond to:
    // Each thread group is 4x4, and GroupID tells us what tile we are
    // GroupThreadID tells us which element in the 4x4 we are
    uint2 pixelCoord = uint2(GroupID.x * 4 + GroupThreadID.x, GroupID.y * 4 + GroupThreadID.y);

    // TODO: The ray for each pixel is constant for a given resolution, fov, and DistToPlane.
    // Might be worth building a lookup texture to use here instead of the math below, though it's
    // not clear whether that's any faster or not, especially since the multiply by CameraWorldTransform
    // has to happen anyways.

    // Compute ray through this pixel (locally, first)
    float3 rayDir = float3((float)pixelCoord.x - HalfSize.x, HalfSize.y - (float)pixelCoord.y, DistToProjPlane);

    // Transform the local ray to the camera's world orientation
    rayDir = mul((float3x3)CameraWorldTransform, normalize(rayDir));

    // Pick the ray start position to be camera's position
    float3 rayStart = CameraWorldTransform._m03_m13_m23;

    if (RayTrace(rayStart, rayDir))
    {
        RenderTarget[pixelCoord] = float4(0, 0.25, 0.75, 1);
    }
    else
    {
        // clear color
        RenderTarget[pixelCoord] = float4(0, 0, 0, 1);
    }
}