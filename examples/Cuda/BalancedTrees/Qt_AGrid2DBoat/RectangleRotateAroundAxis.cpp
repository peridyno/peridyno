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

		Real m_angle = this->varInitialAngle()->getData();
		updateCenterAndRotation(m_angle);
	}

	template<typename TDataType>
	void RectangleRotateAroundAxis<TDataType>::updateCenterAndRotation(Real y_rotation)
	{
		this->varVisPlane()->setValue(1);
		this->varRotation2D()->setValue(y_rotation);

		//Adjust the size of the bounding box
		Coord3D m_center = (m_alignedBox.v0 + m_alignedBox.v1) / 2;
		Coord3D m_length = m_alignedBox.v1 - m_alignedBox.v0;
		Coord2D center2D = this->computeRotate(Coord2D(m_center[0] + this->varRotationRadius()->getData(), m_center[2] - m_length[2] * 1.5f / 2.0f));

		this->varWidth()->setValue(m_length[0] * 3.5f, false);
		this->varHeight()->setValue(m_length[2] * 2.5f, false);
		this->varLocation2D()->setValue(center2D);
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