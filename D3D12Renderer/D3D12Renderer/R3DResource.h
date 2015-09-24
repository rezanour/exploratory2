#pragma once

// Base class for all graphics resources

class R3DResource
{
public:
    virtual ~R3DResource();

    // Is the resource resident?
    bool IsResident() const;

    // Is the resource still referenced?
    bool IsReferenced() const;

    // Make the resource resident (WARNING: blocking call!)
    HRESULT MakeResident();

    // Evict the resource from memory (WARNING: blocking call if resource in use!)
    void Evict();

    // After submitting work that depends on the resource, call this to track that reference
    void ReferenceResource(const ComPtr<ID3D12CommandQueue>& queue);

protected:
    R3DResource(const ComPtr<ID3D11Pageable>& resource, const ComPtr<ID3D12Fence>& fence);

private:
    ComPtr<ID3D11Pageable> Resource;
    ComPtr<ID3D12Fence> Fence;
    // Resource still in use until fence value crossed
    uint64_t LastReferencedFenceValue;
    bool Resident;
};
