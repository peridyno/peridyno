#include "LandScape.h"

#include "Topology/HeightField.h"

#include "Mapping/HeightFieldToTriangleSet.h"
#include "GLSurfaceVisualModule.h"

#define STB_IMAGE_IMPLEMENTATION
#include <image/stb_image.h>

namespace dyno
{ 
    template<typename TDataType>
    LandScape<TDataType>::LandScape()
        : ParametricModel<TDataType>()
    {
        auto heights = std::make_shared<HeightField<TDataType>>();
        this->stateHeightField()->setDataPtr(heights);

        auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
        //mapper->varTranslation()->setValue(Vec3f(-128.0f, -5.0f, -128.0f));
        this->stateHeightField()->promoteOuput()->connect(mapper->inHeightField());
        this->graphicsPipeline()->pushModule(mapper);

        auto sRender = std::make_shared<GLSurfaceVisualModule>();
        //sRender->setColor(Color(0.57f, 0.4f, 0.3f));
        sRender->varUseVertexNormal()->setValue(true);
        mapper->outTriangleSet()->connect(sRender->inTriangleSet());
        this->graphicsPipeline()->pushModule(sRender);

        auto callback = std::make_shared<FCallBackFunc>(std::bind(&LandScape<TDataType>::callbackTransform, this));

        this->varLocation()->attach(callback);
        this->varScale()->attach(callback);
        this->varRotation()->attach(callback);

        auto callbackLoadFile = std::make_shared<FCallBackFunc>(std::bind(&LandScape<TDataType>::callbackLoadFile, this));
        this->varFileName()->attach(callbackLoadFile);
    }

    template<typename TDataType>
    LandScape<TDataType>::~LandScape()
    {
    }

    template<typename TDataType>
    void LandScape<TDataType>::resetStates()
    {
        callbackTransform();
    }

    template<typename Real, typename Coord>
    __global__ void LS_Transform(
        DArray2D<Coord> disp,
        DArray2D<Real> heights,
		Coord scale)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

        uint nx = heights.nx();
        uint ny = heights.ny();
        if (i < nx && j < ny)
        {
            Real y = heights(i, j);

            disp(i, j) = Coord(0, y * scale.y, 0);
        }
    }

    template<typename TDataType>
    void LandScape<TDataType>::callbackTransform()
    {
        auto scale = this->varScale()->getValue();
        auto loc = this->varLocation()->getValue();

        auto topo = this->stateHeightField()->getDataPtr();

        uint nx = mInitialHeights.nx();
        uint nz = mInitialHeights.ny();

        topo->setExtents(mInitialHeights.nx(), mInitialHeights.ny());
        topo->setGridSpacing(1);
        topo->setOrigin(Coord(-0.5 * nx * scale.x + loc.x, loc.y, -0.5 * nz * scale.z + loc.z));

        auto& disp = topo->getDisplacement();

        cuExecute2D(make_uint2(mInitialHeights.nx(), mInitialHeights.ny()),
            LS_Transform,
            disp,
            mInitialHeights,
            scale);
    }

	template<typename TDataType>
	void LandScape<TDataType>::callbackLoadFile()
	{
		const std::string& mapPath = this->varFileName()->getValue().string();
		if (mapPath != fileName) {
			fileName = mapPath;

			int w, h, comp;
			stbi_set_flip_vertically_on_load(true);

			float* data = stbi_loadf(fileName.c_str(), &w, &h, &comp, STBI_default);

			CArray2D<Real> hLand(w, h);

			for (int x0 = 0; x0 < w; x0++)
			{
				for (int y0 = 0; y0 < h; y0++)
				{
					int idx = (y0 * w + x0) * comp;

					hLand(x0, y0) = data[idx];
				}
			}

            mInitialHeights.assign(hLand);

			delete data;
		}
	}
    
    DEFINE_CLASS(LandScape);
}