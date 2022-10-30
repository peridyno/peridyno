#ifndef FLUID3D_SHARED_H
#define FLUID3D_SHARED_H

#define DEF_H 1             //grid spacing
#define DEF_OMEGA 1         //surface width

#define FLUID_ALPHA 0.7    //parameter for density diffusion
#define FLUID_BETA 1     //parameter for sharpening

#define INTER_WEIGHT 0.5

struct GridInfo
{
    float spacing;

    uint nx;
    uint ny;
    uint nz;

    float ox;
    float oy;
    float oz;
};

struct Array3DInfo
{
    uint nx;
    uint ny;
    uint nz;
    uint nxy;
};

struct Coefficent
{
	float a;
	float x0;
	float x1;
	float y0;
	float y1;
	float z0;
	float z1;
	float padding;
};

bool inside(uint i, uint j, uint k, Array3DInfo info)
{
    return i < info.nx && j < info.ny && k < info.nz;
}

//TODO: the macro definition "#define INDEX(i, j, k, aInfo) i + j * aInfo.nx + k * aInfo.nxy" is not supported in vulkan
uint INDEX(uint i, uint j, uint k, Array3DInfo aInfo)
{
    return i + j * aInfo.nx + k * aInfo.nxy;
}

float SharpeningWeight(float dist)
{
    return 1.0;
    float fx = dist - floor(dist);
    fx = 1.0f - 2.0f*abs(fx - 0.5);

    if (fx < 0.01f)
    {
        fx = 0.0f;
    }

    return fx;
}

#define RHO_SOLID 2000

#define RHO_LIQUID 1000
#define RHO_AIR 10

float CalculateBoundaryFraction(float s, float h)
{
    return clamp(0.5*s/h, -0.5, 0.5) + 0.5;
}

float CalculateDensity(float liquidFrac, float solidFrac)
{
    float scale = liquidFrac * min(liquidFrac + solidFrac, 1);
    return (1-scale)*(RHO_LIQUID*liquidFrac + RHO_AIR*(1.0f - liquidFrac)) + scale * RHO_SOLID;
}


#endif