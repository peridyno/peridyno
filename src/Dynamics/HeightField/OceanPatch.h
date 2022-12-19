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
#include <cufft.h>
#include <vector>
#include "Node.h"

#include "Complex.h"

namespace dyno {

    struct WindParam
    {
        float windSpeed;
        float A;
        float choppiness;
        float global;
    };

    template<typename TDataType>
    class OceanPatch : public Node
    {
        DECLARE_TCLASS(OceanPatch, TDataType)
    public:
        typedef typename TDataType::Real Real;
        typedef typename TDataType::Coord Coord;
        typedef typename Complex<Real> Complex;

        OceanPatch();
        ~OceanPatch();

    public:
		DEF_VAR(uint, WindType, 2, "wind Types");//风速等级

        DEF_VAR(Real, WindDirection, Real(60), "Wind direction");

        DEF_VAR(uint, Resolution, 512, "");

        DEF_VAR(Real, PatchSize, Real(512), "Real patch size");

        DEF_VAR(Real, TimeScale, Real(1), "");

    public:
		DEF_ARRAY2D_STATE(Coord, Displacement, DeviceType::GPU, "");

		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");

    protected:
        void resetStates() override;

        void updateStates() override;
        void postUpdateStates() override;

    private:
        void generateH0(Complex* h0);
        void resetWindType();

        std::vector<WindParam> mParams;  //A set of pre-defined configurations

		DArray2D<Complex> mH0;  //初始频谱
		DArray2D<Complex> mHt;  //当前时刻频谱

		DArray2D<Complex> mDxt;  //x方向偏移
		DArray2D<Complex> mDzt;  //z方向偏移

        const Real g = 9.81f;          //重力
        
        Real mDirDepend = 0.07f;  //风长方向相关性

        Real mAmplitude = 1e-7f;          //波的缩放系数
        Real mWindSpeed = 1.0f;
        Real mChoppiness;  //设置浪尖的尖锐性，范围0~1
        Real mGlobalShift;    //大尺度偏移幅度

        cufftHandle fftPlan;

		int mSpectrumWidth;  //频谱宽度
		int mSpectrumHeight;  //频谱长度
    };

	IMPLEMENT_TCLASS(OceanPatch, TDataType)
}
