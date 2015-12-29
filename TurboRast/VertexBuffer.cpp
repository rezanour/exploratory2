#include "Precomp.h"
#include "VertexBuffer.h"

VertexBuffer::VertexBuffer()
{
}

VertexBuffer::~VertexBuffer()
{
}

void VertexBuffer::Update(const Vertex* vertices, uint64_t numVertices)
{
    NumVertices = numVertices;

    // Restructure input data into SSE friendly layout (AOSOA)
    Blocks.clear();
    for (uint64_t i = 0; i < numVertices; i += 4)
    {
        Blocks.push_back(SSEVertexBlock(&vertices[i], std::min(numVertices - i, 4ull)));
    }
}
