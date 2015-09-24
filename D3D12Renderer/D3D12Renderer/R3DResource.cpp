#include "Precomp.h"
#include "R3DResource.h"
#include "Errors.h"

R3DResource::R3DResource(const ComPtr<ID3D12Pageable>& resource, const ComPtr<ID3D12Fence>& fence)
    : Resource(resource)
    , Fence(fence)
    , LastReferencedFenceValue(0)
    , Resident(true)
{
}

R3DResource::~R3DResource()
{
    Evict();
}

bool R3DResource::IsResident() const
{
    return Resident;
}

bool R3DResource::IsReferenced() const
{
    return (Fence->GetCompletedValue() < LastReferencedFenceValue);
}

HRESULT R3DResource::MakeResident()
{
    ComPtr<ID3D12Device> device;
    resource->GetDevice(&device);
    CHECKHR(device->MakeResident(1, resource.GetAddressOf()));
}

void R3DResource::Evict()
{
    Fence->Wait(LastReferencedFenceValue);

    ComPtr<ID3D12Device> device;
    resource->GetDevice(&device);
    CHECKHR(device->Evict(1, resource.GetAddressOf()));
}

void R3DResource::ReferenceResource(const ComPtr<ID3D12CommandQueue>& queue)
{
    // TODO: Need to wrap up the fence object with something else so we can get the right next value.
    // This is wrong
    LastReferencedFenceValue = Fence->GetCompletedValue();

    queue->Signal(Fence, LastReferencedFenceValue);
}
