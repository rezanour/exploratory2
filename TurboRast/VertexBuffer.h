#pragma once

class VertexBuffer
{
public:
    VertexBuffer();
    ~VertexBuffer();

    void Update(const Vertex* vertices, uint64_t numVertices);

    const SSEVertexBlock* const GetBlocks() const { return Blocks.data(); }
    uint64_t GetNumBlocks() const { return (uint64_t)Blocks.size(); }
    uint64_t GetNumVertices() const { return NumVertices; }

private:
    VertexBuffer(const VertexBuffer&) = delete;
    VertexBuffer& operator= (const VertexBuffer&) = delete;

private:
    std::vector<SSEVertexBlock> Blocks;
    uint64_t NumVertices = 0;
};
