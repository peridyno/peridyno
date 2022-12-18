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
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <math_constants.h>
#include "Node.h"

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
        typedef typename Vector<float, 2> Coord;

        OceanPatch(std::string name = "default");
        OceanPatch(int size, float patchSize, int windType = 1, std::string name = "default");
        OceanPatch(int size, float wind_dir, float windSpeed, float A_p, float max_choppiness, float global);
        ~OceanPatch();

        void animate(float t);

        float getMaxChoppiness();
        float getChoppiness();

        //返回实际覆盖面积，以m为单位
        float getPatchSize()
        {
            return m_realPatchSize;
        }

        //返回网格分辨率
        float getGridSize()
        {
            return mResolution;
        }
        float getGlobalShift()
        {
            return m_globalShift;
        }
        float getGridLength()
        {
            return m_realPatchSize / mResolution;
        }
        void setChoppiness(float value)
        {
            mChoppiness = value;
        }

        DArray2D<Coord> getHeightField()
        {
            return m_ht;
        }
        DArray2D <Vec4f> getDisplacement()
        {
            return m_displacement;
        }

    public:
        //float m_windSpeed = 4;                   //风速
        float windDir = CUDART_PI_F / 3.0f;  //风场方向
        int   m_windType;                        //风力等级，目前设置为0~12
        float m_fft_real_length = 10;
        float m_fft_flow_speed = 1.0f;

        DArray2D<Vec4f> m_displacement;  // 位移场
        DArray2D<Vec4f> m_gradient;      // gradient field

        DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");

    protected:
        void resetStates() override;

        void updateStates() override;
        void updateTopology() override;

    private:
        void  generateH0(Coord* h0);
        float gauss();
        float phillips(float Kx, float Ky, float Vdir, float V, float A, float dir_depend);
        void resetWindType();

        int mResolution;

        int mSpectrumWidth;  //频谱宽度
        int mSpectrumHeight;  //频谱长度

        float mChoppiness;  //设置浪尖的尖锐性，范围0~1

        std::vector<WindParam> m_params;  //不同风力等级下的FFT变换参数

        const float g = 9.81f;          //重力
        float       A = 1e-7f;          //波的缩放系数
        float       m_realPatchSize;    //实际覆盖面积，以m为单位
        float       dirDepend = 0.07f;  //风长方向相关性

        float m_maxChoppiness;  //设置choppiness上限
        float m_globalShift;    //大尺度偏移幅度

        DArray2D<Coord> m_h0;  //初始频谱
        DArray2D<Coord> m_ht;  //当前时刻频谱

        DArray2D<Coord> m_Dxt;  //x方向偏移
        DArray2D<Coord> m_Dzt;  //z方向偏移

        cufftHandle fftPlan;

        DEF_VAR(int, WindType, 4, "wind Types");//风速等级
        DEF_VAR(float, WindSpeed, 4, "wind Speed");//风速等级

        //DEF_VAR(int, Size, 512, "size");
        //DEF_VAR(float, PatchSize, 512, "Patch Size");

    };
    IMPLEMENT_TCLASS(OceanPatch, TDataType)
}
