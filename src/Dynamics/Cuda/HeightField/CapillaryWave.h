#pragma once
#include "Node.h"

namespace dyno
{
	/*!
	*	\class	CapillaryWave
	*	\brief	Peridynamics-based CapillaryWave.
	*/
	template<typename TDataType>
	class CapillaryWave : public Node
	{
		DECLARE_TCLASS(CapillaryWave, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename dyno::Vector<float, 4> Coord4;

		CapillaryWave(int size, float patchLength, std::string name = "capillaryWave");
		CapillaryWave(std::string name = "capillaryWave");

		virtual ~CapillaryWave();

		DArray2D<Coord4> GetHeight() { return mHeight; }

		float getHorizon() { return horizon; }

		void setOriginX(int x) { simulatedOriginX = x; }
		void setOriginY(int y) { simulatedOriginY = y; }

		int simulatedOriginX = 0;			//��̬�����ʼx����
		int simulatedOriginY = 0;			//��̬�����ʼy����

		int getOriginX() { return simulatedOriginX; }
		int getOriginZ() { return simulatedOriginY; }

		int getGridSize() { return simulatedRegionWidth; }
		float getRealGridSize() { return realGridSize; }

		DArray2D<Coord4> getHeightField() { return mHeight; }
		Vec2f getOrigin() { return Vec2f(simulatedOriginX * realGridSize, simulatedOriginY * realGridSize); }
		
		void addSource();
		void moveDynamicRegion(int nx, int ny);		//���洬���ƶ���̬��������
	
		DArray2D<Vec2f> getmSource() { return mSource; }
		DArray2D<float> getWeight() { return mWeight; }
		
		void compute();
		void updateStates() override;

		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");

	protected:
		void resetStates() override;




		void updateTopology() override;

		void initialize();
		void initDynamicRegion();
		void initSource();
		void resetSource();
		void swapDeviceGrid();
		void initHeightPosition();

	public:
		int mResolution;

		float mChoppiness;  //�����˼�ļ����ԣ���Χ0~1
	
	protected:
		float patchLength;
		float realGridSize;

		float simulatedRegionWidth;
		float simulatedRegionHeight;

		size_t gridPitch;

		float horizon = 2.0f;			        //ˮ���ʼ�߶�

		DArray2D<Coord4> mHeight;				//�߶ȳ�
		DArray2D<Coord4> mDeviceGrid;		    //��ǰ��̬����״̬
		DArray2D<Coord4> mDeviceGridNext;
		DArray2D<Coord4> mDisplacement;         // λ�Ƴ�
		DArray2D<Vec2f> mSource;				//�������Ӵ���ˮ����
		DArray2D<float> mWeight;				
	};

	IMPLEMENT_TCLASS(CapillaryWave, TDataType)
}