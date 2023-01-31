#pragma once
#include "Module/TopologyModule.h"

namespace dyno
{
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

	class UniformGrid3D : public dyno::TopologyModule
	{
	public:
		UniformGrid3D();
		~UniformGrid3D();

		float spacing() { return mSpacing; }

		dyno::Vec3f orgin() { return mOrigin; }

		uint32_t nx() { return mDimension.x; }
		uint32_t ny() { return mDimension.y; }
		uint32_t nz() { return mDimension.z; }

		uint32_t totalGridSize() {
			return mDimension.x * mDimension.y * mDimension.z;
		}

		void setNx(uint32_t nx) { mDimension.x = nx; }
		void setNy(uint32_t ny) { mDimension.y = ny; }
		void setNz(uint32_t nz) { mDimension.z = nz; }

		void setSpacing(float s) { mSpacing = s; }
		void setOrigin(dyno::Vec3f pos) { mOrigin = pos; }

		GridInfo getGridInfo();

	private:
		float mSpacing;

		dyno::Vec3f mOrigin;
		dyno::Vec3u mDimension;
	};
}


