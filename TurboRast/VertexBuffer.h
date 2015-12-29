#pragma once

class TRVertexBuffer
{
    NON_COPYABLE(TRVertexBuffer);

public:
    TRVertexBuffer();
    ~TRVertexBuffer();

    void Update(const Vertex* vertices, uint64_t numVertices);

    const SSEVertexBlock* const GetBlocks() const { return Blocks.data(); }
    uint64_t GetNumBlocks() const { return (uint64_t)Blocks.size(); }
    uint64_t GetNumVertices() const { return NumVertices; }

private:
    std::vector<SSEVertexBlock> Blocks;
    uint64_t NumVertices = 0;
};
