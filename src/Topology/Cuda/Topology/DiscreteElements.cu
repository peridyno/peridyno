#include "DiscreteElements.h"

namespace dyno
{
	IMPLEMENT_TCLASS(DiscreteElements, TDataType)

	template<typename TDataType>
	DiscreteElements<TDataType>::DiscreteElements()
		: TopologyModule()
	{
	}

	template<typename TDataType>
	DiscreteElements<TDataType>::~DiscreteElements()
	{
// 		m_hostBoxes.clear();
// 		m_hostSpheres.clear();
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::scale(Real s)
	{
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::totalSize()
	{
		return m_boxes.size() + m_spheres.size() + m_tets.size() + m_caps.size() + m_tris.size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::totalJointSize()
	{
		return mBallAndSocketJoints.size() + mSliderJoints.size() + mHingeJoints.size() + mFixedJoints.size() + mPointJoints.size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::sphereIndex()
	{
		return 0;
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::triangleIndex()
	{
		return capsuleIndex() + this->getCaps().size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::tetIndex()
	{
		return boxIndex() + this->getBoxes().size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::capsuleIndex()
	{
		return tetIndex() + this->getTets().size();
	}

	template<typename TDataType>
	uint DiscreteElements<TDataType>::boxIndex()
	{
		return sphereIndex() + this->getSpheres().size();
	}

	template<typename TDataType>
	ElementOffset DiscreteElements<TDataType>::calculateElementOffset()
	{
		ElementOffset elementOffset;
		elementOffset.setSphereRange(sphereIndex(), sphereIndex() + this->getSpheres().size());
		elementOffset.setBoxRange(boxIndex(), boxIndex() + this->getBoxes().size());
		elementOffset.setTetRange(tetIndex(), tetIndex() + this->getTets().size());
		elementOffset.setCapsuleRange(capsuleIndex(), capsuleIndex() + this->getCaps().size());
		elementOffset.setTriangleRange(triangleIndex(), triangleIndex() + this->getTris().size());

		return elementOffset;
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setBoxes(DArray<Box3D>& boxes)
	{
		m_boxes.assign(boxes);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setSpheres(DArray<Sphere3D>& spheres)
	{
		m_spheres.assign(spheres);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTetSDF(DArray<Real>& sdf)
	{
		m_tet_sdf.assign(sdf);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTets(DArray<Tet3D>& tets)
	{
		m_tets.assign(tets);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setCapsules(DArray<Capsule3D>& capsules)
	{
		m_caps.assign(capsules);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTetBodyId(DArray<int>& body_id)
	{
		m_tet_body_mapping.assign(body_id);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTetElementId(DArray<TopologyModule::Tetrahedron>& element_id)
	{
		m_tet_element_id.assign(element_id);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::setTriangles(DArray<Triangle3D>& triangles)
	{
		m_tris.assign(triangles);
	}

	template<typename TDataType>
	void dyno::DiscreteElements<TDataType>::copyFrom(DiscreteElements<TDataType>& de)
	{
		m_spheres.assign(de.m_spheres);
		m_boxes.assign(de.m_boxes);
		m_tets.assign(de.m_tets);
		m_caps.assign(de.m_caps);
		m_tris.assign(de.m_tris);

		m_tet_sdf.assign(de.m_tet_sdf);
		m_tet_body_mapping.assign(de.m_tet_body_mapping);
		m_tet_element_id.assign(de.m_tet_element_id);
	}

	template<typename Coord, typename Matrix, typename Box3D>
	__global__ void DE_Local2Global(
		DArray<Box3D> boxInGlobal,
		DArray<Sphere3D> sphereInGlobal,
		DArray<Tet3D> tetInGlobal,
		DArray<Capsule3D> capInGlobal,
		DArray<Box3D> boxInLocal,
		DArray<Sphere3D> sphereInLocal,
		DArray<Tet3D> tetInLocal,
		DArray<Capsule3D> capInLocal,
		DArray<Coord> positionGlobal,
		DArray<Matrix> rotationGlobal,
		ElementOffset elementOffset)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= rotationGlobal.size()) return;

		ElementType eleType = elementOffset.checkElementType(tId);

		Coord t = positionGlobal[tId];
		Matrix r = rotationGlobal[tId];

		switch (eleType)
		{
		case ET_SPHERE:
		{
			sphereInGlobal[tId - elementOffset.sphereIndex()] = local2Global(sphereInLocal[tId - elementOffset.sphereIndex()], t, r);
			break;
		}
		case ET_BOX:
		{
			boxInGlobal[tId - elementOffset.boxIndex()] = local2Global(boxInLocal[tId - elementOffset.boxIndex()], t, r);
			break;
		}
		case ET_TET:
		{
			tetInGlobal[tId - elementOffset.tetIndex()] = local2Global(tetInLocal[tId - elementOffset.tetIndex()], t, r);
			break;
		}
		case ET_CAPSULE:
		{
			capInGlobal[tId - elementOffset.capsuleIndex()] = local2Global(capInLocal[tId - elementOffset.capsuleIndex()], t, r);
			break;
		}
		// 		case ET_TRI:
		// 		{
		// 			//TODO:
		// // 			boundary_expand = 0.01;
		// // 			box = tris[tId - elementOffset.triangleIndex()].aabb();
		// 			break;
		// 		}
		default:
			break;
		}
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::requestDiscreteElementsInGlobal(
		DArray<Box3D>& boxInGlobal, 
		DArray<Sphere3D>& sphereInGlobal, 
		DArray<Tet3D>& tetInGlobal, 
		DArray<Capsule3D>& capInGlobal)
	{
		auto elementOffset = this->calculateElementOffset();

		boxInGlobal.assign(this->getBoxes());
		sphereInGlobal.assign(this->getSpheres());
		tetInGlobal.assign(this->getTets());
		capInGlobal.assign(this->getCaps());

		cuExecute(this->totalSize(),
			DE_Local2Global,
			boxInGlobal,
			sphereInGlobal,
			tetInGlobal,
			capInGlobal,
			this->getBoxes(),
			this->getSpheres(),
			this->getTets(),
			this->getCaps(),
			mPosition,
			mRotation,
			elementOffset);
	}

	template<typename Coord, typename Matrix, typename Box3D>
	__global__ void DE_Local2GlobalForBox(
		DArray<Box3D> boxInGlobal,
		DArray<Box3D> boxInLocal,
		DArray<Coord> positionGlobal,
		DArray<Matrix> rotationGlobal,
		uint offset)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boxInLocal.size()) return;

		Coord t = positionGlobal[tId + offset];
		Matrix r = rotationGlobal[tId + offset];

		boxInGlobal[tId] = local2Global(boxInLocal[tId], t, r);
	}

	template<typename Coord, typename Matrix, typename Sphere3D>
	__global__ void DE_Local2GlobalForSphere(
		DArray<Sphere3D> sphereInGlobal,
		DArray<Sphere3D> sphereInLocal,
		DArray<Coord> positionGlobal,
		DArray<Matrix> rotationGlobal,
		uint offset)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= sphereInLocal.size()) return;

		Coord t = positionGlobal[tId + offset];
		Matrix r = rotationGlobal[tId + offset];

		sphereInGlobal[tId] = local2Global(sphereInLocal[tId], t, r);
	}

	template<typename Coord, typename Matrix, typename Tet3D>
	__global__ void DE_Local2GlobalForTet(
		DArray<Tet3D> tetInGlobal,
		DArray<Tet3D> tetInLocal,
		DArray<Coord> positionGlobal,
		DArray<Matrix> rotationGlobal,
		uint offset)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tetInLocal.size()) return;

		Coord t = positionGlobal[tId + offset];
		Matrix r = rotationGlobal[tId + offset];

		tetInGlobal[tId] = local2Global(tetInLocal[tId], t, r);
	}

	template<typename Coord, typename Matrix, typename Capsule3D>
	__global__ void DE_Local2GlobalForCapsule(
		DArray<Capsule3D> capInGlobal,
		DArray<Capsule3D> capInLocal,
		DArray<Coord> positionGlobal,
		DArray<Matrix> rotationGlobal,
		uint offset)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= capInLocal.size()) return;

		Coord t = positionGlobal[tId + offset];
		Matrix r = rotationGlobal[tId + offset];

		capInGlobal[tId] = local2Global(capInLocal[tId], t, r);
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::requestCapsuleInGlobal(DArray<Capsule3D>& capInGlobal)
	{
		auto elementOffset = this->calculateElementOffset();

		capInGlobal.assign(this->getCaps());

		cuExecute(capInGlobal.size(),
			DE_Local2GlobalForCapsule,
			capInGlobal,
			this->getCaps(),
			mPosition,
			mRotation,
			elementOffset.capsuleIndex());
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::requestTetInGlobal(DArray<Tet3D>& tetInGlobal)
	{
		auto elementOffset = this->calculateElementOffset();

		tetInGlobal.assign(this->getTets());

		cuExecute(tetInGlobal.size(),
			DE_Local2GlobalForCapsule,
			tetInGlobal,
			this->getTets(),
			mPosition,
			mRotation,
			elementOffset.tetIndex());
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::requestSphereInGlobal(DArray<Sphere3D>& sphereInGlobal)
	{
		auto elementOffset = this->calculateElementOffset();

		sphereInGlobal.assign(this->getSpheres());

		cuExecute(sphereInGlobal.size(),
			DE_Local2GlobalForSphere,
			sphereInGlobal,
			this->getSpheres(),
			mPosition,
			mRotation,
			elementOffset.sphereIndex());
	}

	template<typename TDataType>
	void DiscreteElements<TDataType>::requestBoxInGlobal(DArray<Box3D>& boxInGlobal)
	{
		auto elementOffset = this->calculateElementOffset();

		boxInGlobal.assign(this->getBoxes());

		cuExecute(boxInGlobal.size(),
			DE_Local2GlobalForBox,
			boxInGlobal,
			this->getBoxes(),
			mPosition,
			mRotation,
			elementOffset.boxIndex());
	}

	DEFINE_CLASS(DiscreteElements);
}