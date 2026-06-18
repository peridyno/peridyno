#pragma once
#include "Topology.h"

#include "Array/Array3D.h"

namespace dyno
{
	template<typename TDataType>
	class UniformGrid3D : public Topology
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		UniformGrid3D();
		~UniformGrid3D() override;

		Coord origin() { return mOrigin; }

		Real gridSpacing() { return mGridSpacing; }

		void setOrigin(Coord o) { mOrigin = o; }
		void setGridSpacing(Real s) { mGridSpacing = s; }

		uint nx() { return mNx; }
		uint ny() { return mNy; }
		uint nz() { return mNz; }

	public:
		virtual void initialize(uint nx, uint ny, uint nz);

	private:
		uint mNx = 0;
		uint mNy = 0;
		uint mNz = 0;

		/**
		 * @brief Lower left corner
		 *
		 */
		Coord mOrigin;

		/**
		 * @brief grid spacing
		 *
		 */
		Real mGridSpacing = 1;
	};
}


