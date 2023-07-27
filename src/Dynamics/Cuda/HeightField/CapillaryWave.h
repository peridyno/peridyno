/**
 * Copyright 2017-2022 hanxingyixue
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "Node.h"
#include "Topology/HeightField.h"
namespace dyno
{
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

		int simulatedOriginX = 0;			
		int simulatedOriginY = 0;			

		int getOriginX() { return simulatedOriginX; }
		int getOriginZ() { return simulatedOriginY; }

		int getGridSize() { return simulatedRegionWidth; }
		float getRealGridSize() { return realGridSize; }

		DArray2D<Coord4> getHeightField() { return mHeight; }
		Vec2f getOrigin() { return Vec2f(simulatedOriginX * realGridSize, simulatedOriginY * realGridSize); }

		void addSource();
		void moveDynamicRegion(int nx, int ny);		

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

		float mChoppiness;  

	protected:
		float patchLength;
		float realGridSize;

		int simulatedRegionWidth;
		int simulatedRegionHeight;

		size_t gridPitch;

		float horizon = 2.0f;			        

		DArray2D<Coord4> mHeight;				
		DArray2D<Coord4> mDeviceGrid;		    
		DArray2D<Coord4> mDeviceGridNext;
		DArray2D<Coord4> mDisplacement;         
		DArray2D<Vec2f> mSource;				
		DArray2D<float> mWeight;

	public:
		DEF_VAR(int, Size, 512, "");
		DEF_VAR(float, PatchLength, 512.0f, "");

		//std::shared_ptr<HeightField<TDataType>> heights;	
	};

	IMPLEMENT_TCLASS(CapillaryWave, TDataType)
}