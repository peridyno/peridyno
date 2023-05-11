#include "CalculateBoundingBox.h"
#include "Primitive/Primitive3D.h"

namespace dyno
{
	IMPLEMENT_TCLASS(CalculateBoundingBox, TDataType)
		
	typedef typename ::dyno::TOrientedBox3D<Real> Box3D;

	template<typename TDataType>
	CalculateBoundingBox<TDataType>::CalculateBoundingBox()
		: ComputeModule()
	{
	}

	template<typename TDataType>
	CalculateBoundingBox<TDataType>::~CalculateBoundingBox()
	{
	}


	template<typename Box3D, typename AABB>
	__global__ void CBB_SetupAABB(
		DArray<AABB> boundingBox,
		DArray<Box3D> boxes,
		DArray<Sphere3D> spheres,
		DArray<Tet3D> tets,
		DArray<Capsule3D> caps,
		DArray<Triangle3D> tris,
		ElementOffset elementOffset,
		Real boundary_expand)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		ElementType eleType = elementOffset.checkElementType(tId);

		boundary_expand = 0.0075f;

		AABB box;
		switch (eleType)
		{
		case ET_SPHERE:
		{
			box = spheres[tId].aabb();

			break;
		}
		case ET_BOX:
		{
			box = boxes[tId - elementOffset.boxIndex()].aabb();
			break;
		}
		case ET_TET:
		{
			
			box = tets[tId - elementOffset.tetIndex()].aabb();
			break;
		}
		case ET_CAPSULE:
		{
			box = caps[tId - elementOffset.capsuleIndex()].aabb();
			break;
		}
		case ET_TRI:
		{
			boundary_expand = 0.01;
			box = tris[tId - elementOffset.triangleIndex()].aabb();
			break;
		}
		default:
			break;
		}

		boundingBox[tId] = box;
	}

	template<typename TDataType>
	void CalculateBoundingBox<TDataType>::compute()
	{
		auto inTopo = this->inDiscreteElements()->getDataPtr();

		if (this->outAABB()->isEmpty())
			this->outAABB()->allocate();

		auto& aabbs = this->outAABB()->getData();

		int num = inTopo->totalSize();

		aabbs.resize(num);

		Real margin = Real(0);

		ElementOffset elementOffset = inTopo->calculateElementOffset();

		cuExecute(num,
			CBB_SetupAABB,
			aabbs,
			inTopo->getBoxes(),
			inTopo->getSpheres(),
			inTopo->getTets(),
			inTopo->getCaps(),
			inTopo->getTris(),
			elementOffset,
			margin);
	}

	DEFINE_CLASS(CalculateBoundingBox);
}