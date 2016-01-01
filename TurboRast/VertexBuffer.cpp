#include "Precomp.h"
#include "VertexBuffer.h"

TRVertexBuffer::TRVertexBuffer()
{
}

TRVertexBuffer::~TRVertexBuffer()
{
}

void TRVertexBuffer::Update(const Vertex* vertices, uint64_t numVertices)
{
    NumVertices = numVertices;

    // Restructure input data into SSE friendly layout (AOSOA)
    Vertices.clear();
    Blocks.clear();

    for (uint64_t i = 0; i < numVertices; ++i)
    {
        Vertices.push_back(vertices[i]);
    }

    for (uint64_t i = 0; i < numVertices; i += 4)
    {
        Blocks.push_back(SSEVertexBlock(&vertices[i], std::min(numVertices - i, 4ull)));
    }
}
