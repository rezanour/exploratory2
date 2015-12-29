#pragma once

class Texture
{
public:
    Texture(void* data, int width, int height, int pitchInPixels)
        : OwnsMemory(false), Data(data), Width(width), Height(height), PitchInPixels(pitchInPixels)
    {
    }

    ~Texture()
    {
        if (OwnsMemory)
        {
            delete [] (uint32_t*)Data;
        }
    }

    const void* GetData() const { return Data; }
    void* GetData() { return Data; }

    int GetWidth() const { return Width; }
    int GetHeight() const { return Height; }
    int GetPitchInPixels() const { return PitchInPixels; }

private:
    Texture(const Texture&) = delete;
    Texture& operator= (const Texture&) = delete;

private:
    bool OwnsMemory;
    void* Data;
    int Width;
    int Height;
    int PitchInPixels;
};
