#pragma once

struct BinnedTriangle
{
    int64_t iNext;
    int64_t indices[3];
    float3 a, b, c;
    float3 ab, bc, ac;
    float3 norm;
};

#pragma warning(push)
#pragma warning(disable: 4201)  // Nameless struct/union
struct OctNode
{
    float3 Min;
    float3 Max;
    float3 Mid; // Only stored as convenience to save computation later. Could be removed

    bool IsLeaf;

    union
    {
        // InnerNode
        int64_t Children[8];

        // LeafNode
        struct
        {
            // Index into index based linked list of triangles
            int64_t iFirstTriangle;
            int64_t NumTriangles;
        };
    };
};

struct KdNode
{
    float3 Min;
    float3 Max;
    uint32_t Axis : 2;
    uint32_t IsLeaf : 1;

    union
    {
        // InnerNode. 0 = front, 1 = back
        struct
        {
            float Value;
            int64_t Children[2];
        };

        // LeafNode
        struct
        {
            // Index into index based linked list of triangles
            int64_t iFirstTriangle;
            int64_t NumTriangles;
        };
    };
};
#pragma warning(pop)

// TurboTrace Device
class TTDevice
{
    NON_COPYABLE(TTDevice);

public:
    TTDevice();
    virtual ~TTDevice();

    bool Initialize(const float3& worldMin, const float3& worldMax);

    void SetFov(float fov);
    void SetRenderTarget(uint32_t* renderTarget, int width, int height, int pitchInPixels);
    void Draw(const float3* vertices, int64_t vertexCount, const matrix4x4& cameraTransform);

private:
    void InitNode(int64_t iNode, int64_t& count, int depth, const float3& min, const float3& max);
    void InitKdNode(int64_t iNode, int64_t& count, int depth, const float3& min, const float3& max);

    void BinTriangle(const float3* verts[3], const int64_t indices[3], int64_t iNode);
    void BinTriangle(const float3* verts[3], const float3 edgeNorms[3], const float3 offsets[3], const int64_t indices[3], int64_t iNode);
    void BinTriangleKd(const float3* verts[3], const int64_t indices[3], int64_t iNode);

    // recursive function, returns true if already hit something
    bool Trace(int x, int y, const float3& start, const float3& dir, int64_t iNode);
    bool TraceKd(int x, int y, const float3& start, const float3& dir, int64_t iNode);

    bool RayTriangleTest(const float3& start, const float3& dir, const BinnedTriangle& triangle);

private:
    static const int TreeDepth = 4;

    matrix4x4 CameraTransform;

    float Fov = 0.f;
    float InvTanFov = 0.f;
    float ZDist = 0.f;

    uint32_t* RenderTarget = nullptr;
    int RTWidth = 0;
    float RTWidthOver2 = 0.f;
    float RTHeightOver2 = 0.f;
    int RTHeight = 0;
    int RTPitch = 0;

    // TODO: vertex data will be arbitrary later,
    // but for bring up, just treat each vertex as just position
    const float3* Vertices = nullptr;
    int64_t VertexCapacity = 0;
    int64_t VertexCount = 0;

    BinnedTriangle* Triangles = nullptr;
    int64_t TriangleCapacity = 0;
    int64_t TriangleCount = 0;

    // Nodes[0] is root
    // Inner nodes are stored in first portion of array
    // Leafs are end, so we can loop through them quicker for clearing
    OctNode* Nodes = nullptr;
    int64_t NodesCapacity = 0;

    KdNode* KdNodes = nullptr;
    int64_t KdNodeCapacity = 0;
};
