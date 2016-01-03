#include "Precomp.h"
#include "TTDevice.h"

TTDevice::TTDevice()
{
}

TTDevice::~TTDevice()
{
    //delete[] Vertices;
    delete[] Triangles;
    delete[] Nodes;
    delete[] KdNodes;
}

bool TTDevice::Initialize(const float3& worldMin, const float3& worldMax)
{
    // We trace in post projection space. In other words:
    //   1. transform incoming triangles
    //   2. bin triangles into [-1, 1] space octtree
    //   3. trace using the octtree

    int depth = TreeDepth;
    while (depth > 0)
    {
        NodesCapacity += (int64_t)pow(8, depth - 1);
        KdNodeCapacity += (int64_t)pow(2, depth - 1);
        --depth;
    }
    Nodes = new OctNode[NodesCapacity];
    KdNodes = new KdNode[KdNodeCapacity];


    int64_t iNode = 0;
    int64_t count = 1;

#if 0
    //float3 min(-1.f, -1.f, 0.f);
    //float3 max(1.f, 1.f, 1.f);
    InitNode(iNode, count, 1, worldMin, worldMax);
#else
    InitKdNode(iNode, count, 1, worldMin, worldMax);
#endif

    return true;
}

void TTDevice::SetFov(float fov)
{
    Fov = fov;
    InvTanFov = 1.f / tanf(fov * 0.5f);

    if (RTWidth > 0)
    {
        ZDist = RTWidthOver2 * InvTanFov;
    }
}

void TTDevice::SetRenderTarget(uint32_t* renderTarget, int width, int height, int pitchInPixels)
{
    RenderTarget = renderTarget;
    RTWidth = width;
    RTHeight = height;
    RTPitch = pitchInPixels;

    RTWidthOver2 = 0.5f * RTWidth;
    RTHeightOver2 = 0.5f * RTHeight;

    if (Fov > 0.f)
    {
        ZDist = RTWidthOver2 * InvTanFov;
    }
}

void TTDevice::Draw(const float3* vertices, int64_t vertexCount, const matrix4x4& cameraTransform)
{
    // Ensure we have enough space for the vertices
    //if (VertexCapacity < vertexCount)
    //{
    //    VertexCapacity = vertexCount;
    //    delete[] Vertices;
    //    Vertices = new float3[VertexCapacity];
    //}
    VertexCapacity = vertexCount;

    // Ensure we have enough room for triangles. Remember that
    // a triangle can be stored in multiple (many) bins, so this
    // needs to be much larger than 3 * vertexCount
    if (TriangleCapacity < VertexCapacity * 3 * 100)
    {
        TriangleCapacity = VertexCapacity * 3 * 100;
        delete[] Triangles;
        Triangles = new BinnedTriangle[TriangleCapacity];
    }

    // Clear counts
    VertexCount = 0;
    TriangleCount = 0;

    // Clear OctTree leaves
    OctNode* node = Nodes;
    for (int64_t i = 0; i < NodesCapacity; ++i, ++node)
    {
        if (node->IsLeaf)
        {
            node->iFirstTriangle = -1;
            node->NumTriangles = 0;
        }
    }

    KdNode* kdNode = KdNodes;
    for (int64_t i = 0; i < KdNodeCapacity; ++i, ++kdNode)
    {
        if (kdNode->IsLeaf)
        {
            kdNode->iFirstTriangle = -1;
            kdNode->NumTriangles = 0;
        }
    }

    // TODO: transform vertices
    Vertices = vertices;
    //memcpy_s(Vertices, VertexCapacity * sizeof(float3), vertices, vertexCount * sizeof(float3));
    VertexCount = vertexCount;

#if 0
    for (int64_t iVertex = 0; iVertex < VertexCount; iVertex += 3)
    {
        const float3* verts[3] = { &Vertices[iVertex], &Vertices[iVertex + 1], &Vertices[iVertex + 2] };
        const int64_t indices[3] = { iVertex, iVertex + 1, iVertex + 2 };

//#define BRUTE_FORCE

#ifdef BRUTE_FORCE
        BinTriangle(verts, indices, 0);
#else
        float3 edges[3] = { *verts[1] - *verts[0], *verts[2] - *verts[1], *verts[0] - *verts[2] };
        float3 norm = cross(edges[0], edges[1]);
        float3 edgeNorms[3] = { cross(edges[0], norm), cross(edges[1], norm), cross(edges[2], norm) };
        float3 offsets[3] =
        {
            float3(edgeNorms[0].x < 0 ? 1.f : 0.f, edgeNorms[0].y < 0 ? 1.f : 0.f, edgeNorms[0].z < 0 ? 1.f : 0.f),
            float3(edgeNorms[1].x < 0 ? 1.f : 0.f, edgeNorms[1].y < 0 ? 1.f : 0.f, edgeNorms[1].z < 0 ? 1.f : 0.f),
            float3(edgeNorms[2].x < 0 ? 1.f : 0.f, edgeNorms[2].y < 0 ? 1.f : 0.f, edgeNorms[2].z < 0 ? 1.f : 0.f),
        };

        BinTriangle(verts, edgeNorms, offsets, indices, 0);
#endif
    }
#endif
    LARGE_INTEGER freq, start, stop;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);

    for (int64_t iVertex = 0; iVertex < VertexCount; iVertex += 3)
    {
        const float3* verts[3] = { &Vertices[iVertex], &Vertices[iVertex + 1], &Vertices[iVertex + 2] };
        const int64_t indices[3] = { iVertex, iVertex + 1, iVertex + 2 };

        BinTriangleKd(verts, indices, 0);
    }

    QueryPerformanceCounter(&stop);

    // Assumes that camera transform is rotation only (no scale)
    matrix4x4 rotation = cameraTransform;
    rotation.r[3] = float4(0, 0, 0, 1);
    float3 position = *(const float3*)&cameraTransform.r[3];

    // Trace
    for (int y = 0; y < RTHeight; ++y)
    {
        for (int x = 0; x < RTWidth; ++x)
        {
            // compute ray
            float3 ray(x - RTWidthOver2, RTHeightOver2 - y, ZDist);
            ray = normalize(ray);

            float4 rotated = mul(rotation, float4(ray, 1.f));
            ray = *(float3*)&rotated;

#if 0
            Trace(x, y, position, ray, 0);
#else
            TraceKd(x, y, position, ray, 0);
#endif
        }
    }

    double elapsedSeconds = (stop.QuadPart - start.QuadPart) / (double)freq.QuadPart;
    wchar_t message[128];
    swprintf_s(message, L"Elapsed: %3.3fms\n", 1000.f * elapsedSeconds);
    OutputDebugString(message);
}

void TTDevice::InitNode(int64_t iNode, int64_t& count, int depth, const float3& min, const float3& max)
{
    OctNode* node = &Nodes[iNode];

    float3 mid = (min + max) * 0.5f;

    node->Min = min;
    node->Mid = mid;
    node->Max = max;
    node->IsLeaf = (depth == TreeDepth);

    if (node->IsLeaf)
    {
        node->iFirstTriangle = -1;
        node->NumTriangles = 0;
    }
    else
    {
        int64_t iChild = count;
        int nextDepth = depth + 1;
        count += 8;     // reserve next 8 consecutive for our children

        InitNode(iChild, count, nextDepth, min, mid);
        node->Children[0] = iChild++;

        InitNode(iChild, count, nextDepth, float3(mid.x, min.y, min.z), float3(max.x, mid.y, mid.z));
        node->Children[1] = iChild++;

        InitNode(iChild, count, nextDepth, float3(min.x, mid.y, min.z), float3(mid.x, max.y, mid.z));
        node->Children[2] = iChild++;

        InitNode(iChild, count, nextDepth, float3(mid.x, mid.y, min.z), float3(max.x, max.y, mid.z));
        node->Children[3] = iChild++;

        InitNode(iChild, count, nextDepth, float3(min.x, min.y, mid.z), float3(mid.x, mid.y, max.z));
        node->Children[4] = iChild++;

        InitNode(iChild, count, nextDepth, float3(mid.x, min.y, mid.z), float3(max.x, mid.y, max.z));
        node->Children[5] = iChild++;

        InitNode(iChild, count, nextDepth, float3(min.x, mid.y, mid.z), float3(mid.x, max.y, max.z));
        node->Children[6] = iChild++;

        InitNode(iChild, count, nextDepth, mid, max);
        node->Children[7] = iChild++;
    }
}

void TTDevice::InitKdNode(int64_t iNode, int64_t& count, int depth, const float3& min, const float3& max)
{
    KdNode* node = &KdNodes[iNode];

    node->Axis = depth % 3;
    node->IsLeaf = (depth == TreeDepth);
    node->Min = min;
    node->Max = max;

    if (node->IsLeaf)
    {
        node->iFirstTriangle = -1;
        node->NumTriangles = 0;
    }
    else
    {
        int64_t iChild = count;
        count += 2;

        int nextDepth = depth + 1;

        float3 mid = (min + max) * 0.5f;

        node->Value = ((float*)&mid.x)[node->Axis];

        float3 newMin = min;
        ((float*)&newMin.x)[node->Axis] = node->Value;

        float3 newMax = max;
        ((float*)&newMax.x)[node->Axis] = node->Value;

        InitKdNode(iChild, count, nextDepth, min, newMax);
        node->Children[0] = iChild++;

        InitKdNode(iChild, count, nextDepth, newMin, max);
        node->Children[1] = iChild++;
    }
}

void TTDevice::BinTriangle(const float3* verts[3], const int64_t indices[3], int64_t iNode)
{
    OctNode* node = &Nodes[iNode];

    if (node->IsLeaf)
    {
        assert(TriangleCount < TriangleCapacity);

        // Append triangle to the triangle list
        BinnedTriangle* triangle = &Triangles[TriangleCount];
        triangle->iNext = node->iFirstTriangle;
        triangle->a = *verts[0];
        triangle->b = *verts[1];
        triangle->c = *verts[2];
        triangle->ab = *verts[1] - *verts[0];
        triangle->bc = *verts[2] - *verts[1];
        triangle->ac = *verts[2] - *verts[0];
        triangle->norm = normalize(cross(triangle->ab, triangle->ac));
        triangle->indices[0] = indices[0];
        triangle->indices[1] = indices[1];
        triangle->indices[2] = indices[2];
        node->iFirstTriangle = TriangleCount++;
        ++node->NumTriangles;
    }
    else
    {
        // if we can cull the triangle out from the node, then 
        // no need to continue
        uint32_t masks[3]{};
        for (int i = 0; i < 3; ++i)
        {
            if (verts[i]->x < node->Min.x) masks[i] |= 0x01;
            if (verts[i]->y < node->Min.y) masks[i] |= 0x02;
            if (verts[i]->z < node->Min.z) masks[i] |= 0x04;
            if (verts[i]->x > node->Max.x) masks[i] |= 0x08;
            if (verts[i]->y > node->Max.y) masks[i] |= 0x10;
            if (verts[i]->z > node->Max.z) masks[i] |= 0x20;
        }

        // if there is any plane for which all 3 failed, reject it
        if (masks[0] & masks[1] & masks[2])
        {
            return;
        }

        // couldn't cull it completely, but we can still ignore
        // any child node that couldn't possibly contain the
        // triangle. We do this by skipping children that
        // are on the opposite side of mid from all 3 verts
        memset(&masks, 0, sizeof(masks));
        for (int i = 0; i < 3; ++i)
        {
            if (verts[i]->x > node->Mid.x) masks[i] |= 0x01;
            if (verts[i]->y > node->Mid.y) masks[i] |= 0x02;
            if (verts[i]->z > node->Mid.z) masks[i] |= 0x04;
        }

        uint32_t any = masks[0] | masks[1] | masks[2];
        for (uint32_t index = 0; index < 8; ++index)
        {
            if (any & index)
            {
                // At least one vertex could be in this sector,
                // pass the triangle on down
                BinTriangle(verts, indices, node->Children[index]);
            }
        }
    }
}

void TTDevice::BinTriangle(const float3* verts[3], const float3 edgeNorms[3], const float3 offsets[3], const int64_t indices[3], int64_t iNode)
{
    OctNode* node = &Nodes[iNode];

    if (node->IsLeaf)
    {
        assert(TriangleCount < TriangleCapacity);

        // Append triangle to the triangle list
        BinnedTriangle* triangle = &Triangles[TriangleCount];
        triangle->iNext = node->iFirstTriangle;
        triangle->a = *verts[0];
        triangle->b = *verts[1];
        triangle->c = *verts[2];
        triangle->ab = *verts[1] - *verts[0];
        triangle->bc = *verts[2] - *verts[1];
        triangle->ac = *verts[2] - *verts[0];
        triangle->norm = normalize(cross(triangle->ab, triangle->ac));
        triangle->indices[0] = indices[0];
        triangle->indices[1] = indices[1];
        triangle->indices[2] = indices[2];
        node->iFirstTriangle = TriangleCount++;
        ++node->NumTriangles;
    }
    else
    {
        for (int iChild = 0; iChild < 8; ++iChild)
        {
            OctNode* child = &Nodes[node->Children[iChild]];
            float3 diff = child->Max - child->Min;

            bool skip = false;
            for (int iEdge = 0; iEdge < 3; ++iEdge)
            {
                float3 trivialReject(diff.x * offsets[iEdge].x, diff.y * offsets[iEdge].y, diff.z * offsets[iEdge].z);
                trivialReject = trivialReject + child->Min;

                if (dot(trivialReject - *verts[iEdge], edgeNorms[iEdge]) < 0)
                {
                    // reject child
                    skip = true;
                    break;
                }
            }

            if (!skip)
            {
                BinTriangle(verts, edgeNorms, offsets, indices, node->Children[iChild]);
            }
        }
    }
}

void TTDevice::BinTriangleKd(const float3* verts[3], const int64_t indices[3], int64_t iNode)
{
    KdNode* node = &KdNodes[iNode];

    if (node->IsLeaf)
    {
        assert(TriangleCount < TriangleCapacity);

        // Append triangle to the triangle list
        BinnedTriangle* triangle = &Triangles[TriangleCount];
        triangle->iNext = node->iFirstTriangle;
        triangle->a = *verts[0];
        triangle->b = *verts[1];
        triangle->c = *verts[2];
        triangle->ab = *verts[1] - *verts[0];
        triangle->bc = *verts[2] - *verts[1];
        triangle->ac = *verts[2] - *verts[0];
        triangle->norm = normalize(cross(triangle->ab, triangle->ac));
        triangle->indices[0] = indices[0];
        triangle->indices[1] = indices[1];
        triangle->indices[2] = indices[2];
        node->iFirstTriangle = TriangleCount++;
        ++node->NumTriangles;
    }
    else
    {
        // if we can cull the triangle out from the node, then 
        // no need to continue
        uint32_t masks[3]{};
        for (int i = 0; i < 3; ++i)
        {
            if (verts[i]->x < node->Min.x) masks[i] |= 0x01;
            if (verts[i]->y < node->Min.y) masks[i] |= 0x02;
            if (verts[i]->z < node->Min.z) masks[i] |= 0x04;
            if (verts[i]->x > node->Max.x) masks[i] |= 0x08;
            if (verts[i]->y > node->Max.y) masks[i] |= 0x10;
            if (verts[i]->z > node->Max.z) masks[i] |= 0x20;
        }

        // if there is any plane for which all 3 failed, reject it
        if (masks[0] & masks[1] & masks[2])
        {
            return;
        }

        float v1 = ((float*)verts[0])[node->Axis];
        float v2 = ((float*)verts[1])[node->Axis];
        float v3 = ((float*)verts[2])[node->Axis];

        if (v1 <= node->Value || v2 <= node->Value || v3 <= node->Value)
        {
            BinTriangleKd(verts, indices, node->Children[0]);
        }
        if (v1 > node->Value || v2 > node->Value || v3 > node->Value)
        {
            BinTriangleKd(verts, indices, node->Children[1]);
        }
    }
}

bool TTDevice::Trace(int x, int y, const float3& start, const float3& dir, int64_t iNode)
{
    OctNode* node = &Nodes[iNode];

    if (node->IsLeaf)
    {
        // TODO: test triangles
        int64_t iTriangle = node->iFirstTriangle;
        for (int64_t i = 0; i < node->NumTriangles; ++i)
        {
            assert(iTriangle >= 0);

            BinnedTriangle* triangle = &Triangles[iTriangle];
            if (RayTriangleTest(start, dir, *triangle))
            {
                RenderTarget[y * RTPitch + x] = 0xFFFF0000;
                return true;
            }

            iTriangle = triangle->iNext;
        }

        return false;
    }
    else
    {
        float3 position = start;

        int steps = 0;
        for (;;)    // won't loop more than 1 or 2 times ever. Verify...
        {
            assert(steps < 10);
            steps++;

            // Start with depth first search for start
            uint32_t index = 0;
            if (position.x > node->Mid.x) index |= 0x01;
            if (position.y > node->Mid.y) index |= 0x02;
            if (position.z > node->Mid.z) index |= 0x04;
            bool hit = Trace(x, y, position, dir, node->Children[index]);

            if (!hit)
            {
                // If on the way back up the tree, we haven't hit anything yet,
                // then trace the ray to see what child it hits next (if any)
                float dists[3]{};

                dists[0] = (dir.x > 0) ? ((index & 0x01) ? node->Max.x - position.x : node->Mid.x - position.x)
                    : ((index & 0x01) ? position.x - node->Mid.x : position.x - node->Min.x);

                dists[1] = (dir.y > 0) ? ((index & 0x02) ? node->Max.y - position.y : node->Mid.y - position.y)
                    : ((index & 0x02) ? position.y - node->Mid.y : position.y - node->Min.y);

                dists[2] = (dir.z > 0) ? ((index & 0x04) ? node->Max.z - position.z : node->Mid.z - position.z)
                    : ((index & 0x04) ? position.z - node->Mid.z : position.z - node->Min.z);

                float3 scaledDirs[3] = { dir * (dists[0] / fabsf(dir.x)), dir * (dists[1] / fabsf(dir.y)), dir * (dists[2] / fabsf(dir.z)) };

                // find axis with minimum length. That's the axis we intersect with next
                int iAxis = 0;
                if (lengthSq(scaledDirs[1]) < lengthSq(scaledDirs[iAxis])) iAxis = 1;
                if (lengthSq(scaledDirs[2]) < lengthSq(scaledDirs[iAxis])) iAxis = 2;

                // lodged in edge, bump it
                if (dists[iAxis] < 0.0001f)
                {
                    position = position + dir * 0.001f;
                }

                // push the point just a bit farther along the line so it clearly crosses
                position = position + scaledDirs[iAxis] * 1.00001f;

                // make sure new position is still inside the node, otherwise we're done
                if (position.x < node->Min.x || position.x > node->Max.x ||
                    position.y < node->Min.y || position.y > node->Max.y ||
                    position.z < node->Min.z || position.z > node->Max.z)
                {
                    return false;
                }
            }
        }
    }

    //return false;
}

bool TTDevice::TraceKd(int x, int y, const float3& start, const float3& dir, int64_t iNode)
{
    KdNode* node = &KdNodes[iNode];

    if (node->IsLeaf)
    {
        int64_t iTriangle = node->iFirstTriangle;
        for (int64_t i = 0; i < node->NumTriangles; ++i)
        {
            assert(iTriangle >= 0);

            BinnedTriangle* triangle = &Triangles[iTriangle];
            if (RayTriangleTest(start, dir, *triangle))
            {
                RenderTarget[y * RTPitch + x] = 0xFFFF0000;
                return true;
            }

            iTriangle = triangle->iNext;
        }

        return false;
    }
    else
    {
        float dist = ((float*)&start.x)[node->Axis] - node->Value;
        if (dist < 0)
        {
            if (TraceKd(x, y, start, dir, node->Children[0]))
            {
                return true;
            }
            else if (((float*)&dir.x)[node->Axis] > 0)
            {
                return TraceKd(x, y, start, dir, node->Children[1]);
            }
        }
        else
        {
            if (TraceKd(x, y, start, dir, node->Children[1]))
            {
                return true;
            }
            else if (((float*)&dir.x)[node->Axis] < 0)
            {
                return TraceKd(x, y, start, dir, node->Children[0]);
            }
        }
    }

    return false;
}

bool TTDevice::RayTriangleTest(const float3& start, const float3& dir, const BinnedTriangle& triangle)
{
    if (dot(triangle.norm, dir) > 0)
    {
        return false;
    }

    float d = dot(start - triangle.a, triangle.norm);
    if (d <= 0)
    {
        return false;
    }

    float cosA = dot(triangle.norm * -1, dir);
    float r = d / cosA;

    // intersection of ray and plane
    float3 p = start + dir * r;

    float3 ap = p - triangle.a;
    float3 wC = cross(triangle.ab, ap);
    if (dot(wC, triangle.norm) < 0)
    {
        return false;
    }

    float3 wB = cross(ap, triangle.ac);
    if (dot(wB, triangle.norm) < 0)
    {
        return false;
    }

    float3 wA = cross(triangle.bc, p - triangle.b);
    if (dot(wA, triangle.norm) < 0)
    {
        return false;
    }

    return true;
}
