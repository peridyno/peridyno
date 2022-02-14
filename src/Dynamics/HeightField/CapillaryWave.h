#pragma once
#include "Node.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <math_constants.h>
#include "types.h"
namespace dyno
{
	/*!
	*	\class	CapillaryWave
	*	\brief	Peridynamics-based CapillaryWave.
	*/
	template<typename TDataType>
	class CapillaryWave : public Node
	{
		DECLARE_CLASS_1(CapillaryWave, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename Vector<float, 4> Coord4;

		CapillaryWave(int size, float patchLength, std::string name = "default");
		virtual ~CapillaryWave();

		DArray2D<Coord4> GetHeight() { return mHeight; }

		float getHorizon() { return horizon; }

		void setOriginX(int x) { simulatedOriginX = x; }
		void setOriginY(int y) { simulatedOriginY = y; }

		int simulatedOriginX = 0;			//动态区域初始x坐标
		int simulatedOriginY = 0;			//动态区域初始y坐标

		int getOriginX() { return simulatedOriginX; }
		int getOriginZ() { return simulatedOriginY; }

		int getGridSize() { return simulatedRegionWidth; }
		float getRealGridSize() { return realGridSize; }

		DArray2D<Coord4> getHeightField() { return mHeight; }
		Vec2f getOrigin() { return Vec2f(simulatedOriginX * realGridSize, simulatedOriginY * realGridSize); }


	protected:
		void resetStates() override;

		void updateStates() override;

		void updateTopology() override;

		void initialize();
		void initDynamicRegion();
		void initSource();
		void resetSource();
		void swapDeviceGrid();
		void compute();
		void initHeightPosition();

	public:
		int mResolution;

		float mChoppiness;  //设置浪尖的尖锐性，范围0~1

	protected:
		float patchLength;
		float realGridSize;

		float simulatedRegionWidth;
		float simulatedRegionHeight;

		size_t gridPitch;

		float horizon = 2.0f;			//水面初始高度
		float* mWeight;

		DArray2D<Coord4> mHeight;				//高度场
		DArray2D<Coord4> mDeviceGrid;		//当前动态区域状态
		DArray2D<Coord4> mDeviceGridNext;
		DArray2D<Coord4> mDisplacement;   // 位移场
		DArray2D<Vec2f> mSource;				//用于添加船与水交互

	private:
		DEF_NODE_PORTS(ParticleEmitter, ParticleEmitter<TDataType>, "Particle Emitters");
	};
}