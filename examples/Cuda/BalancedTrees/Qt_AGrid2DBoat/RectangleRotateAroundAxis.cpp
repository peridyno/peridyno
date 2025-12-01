#include "RectangleRotateAroundAxis.h"

namespace dyno
{
	template<typename TDataType>
	RectangleRotateAroundAxis<TDataType>::RectangleRotateAroundAxis()
		: RectangleModel2D<TDataType>()
	{
	}

	template<typename TDataType>
	void RectangleRotateAroundAxis<TDataType>::resetStates()
	{
		int shapenum = this->inTextureMesh()->getDataPtr()->shapes().size();
		if (shapenum == 0) printf("Please check the input TextureMesh!  \n");

		m_alignedBox = this->inTextureMesh()->getDataPtr()->shapes()[0]->boundingBox;
		for (int i = 1; i < shapenum; i++)
		{
			auto bounding = this->inTextureMesh()->getDataPtr()->shapes()[i]->boundingBox;
			m_alignedBox = m_alignedBox.merge(bounding);
		}

		//Adjust the size of the bounding box
		m_center = (m_alignedBox.v0 + m_alignedBox.v1) / 2;
		Coord3D m_length = m_alignedBox.v1 - m_alignedBox.v0;
		this->varLength()->setValue(Coord3D(m_length[0], 0.0f, m_length[2]));
		this->varLocation()->setValue(Coord3D(m_center[0], 0.0f, m_center[2]));

		Real m_angle = this->varInitialAngle()->getData();
		updateCenterAndRotation(m_angle);

	}

	template<typename TDataType>
	void RectangleRotateAroundAxis<TDataType>::updateCenterAndRotation(Real y_rotation)
	{
		this->varRotation()->setValue(Coord3D(0.0f, y_rotation, 0.0f));

		Quat<Real> q = this->computeQuaternion();
		q.normalize();

		Coord3D length = this->varLength()->getData();
		Coord3D center = q.rotate(Coord3D(m_center[0] + this->varRotationRadius()->getData(), 0.0f, m_center[2]- length[2] * 1.5f / 2.0f));
		Coord3D u_axis = q.rotate(Coord3D(1, 0, 0));
		Coord3D w_axis = q.rotate(Coord3D(0, 0, 1));

		TOrientedBox2D<Real> retangle;
		retangle.center = Coord2D(center[0], center[2]);
		retangle.u = Coord2D(u_axis[0], u_axis[2]);
		retangle.v = Coord2D(w_axis[0], w_axis[2]);
		retangle.extent = Coord2D(length[0] * 3.5f / 2.0f, length[2] * 2.5f / 2.0f);

		this->outRectangle()->setValue(retangle);
	}


	template<typename TDataType>
	void RectangleRotateAroundAxis<TDataType>::updateStates()
	{
		uint m_frame = this->stateFrameNumber()->getValue();
		auto m_freq = this->varFrequency()->getData();

		Real m_angle =this->varInitialAngle()->getData(); 
		m_angle -= m_frame * 360.0f / m_freq;
		updateCenterAndRotation(m_angle);
	}

	DEFINE_CLASS(RectangleRotateAroundAxis);
}