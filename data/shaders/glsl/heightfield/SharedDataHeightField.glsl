#define CUDART_PI_F 3.141592654
#define EPSILON 0.000001
#define GRAVITY 9.83219 * 0.5
#define M_PI 3.14159265358979323846
struct Array2DInfo
{
    uint nx;
    uint ny;
};



struct WaveParameters {
	float          choppiness;
	float          realGridSize;
	float          virtualGridSize;
	float          globalShift;
};

uint INDEX(uint i, uint j,  Array2DInfo aInfo)
{
    return i + j * aInfo.nx;
}

bool inside(uint i, uint j,  Array2DInfo info)
{
    return i < info.nx && j < info.ny;
}