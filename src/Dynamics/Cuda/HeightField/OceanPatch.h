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
#include "Topology/HeightField.h"

namespace dyno {

    struct WindParam
    {
        float windSpeed;
        float A;
        float choppiness;
        float global;
    };

    template<typename TDataType>
    class OceanPatch : virtual public Node
    {
        DECLARE_TCLASS(OceanPatch, TDataType)
    public:
        typedef typename TDataType::Real Real;
        typedef typename TDataType::Coord Coord;
        typedef typename ::dyno::Complex<Real> Complex;

        OceanPatch();
        ~OceanPatch();

        std::string getNodeType() override { return "Height Fields"; }
    public:
        DEF_VAR(uint, WindType, 2, "wind Types");

        DEF_VAR(Real, Amplitude, 0, "");

        DEF_VAR(Real, AmplitudeScale, 1, "");

        DEF_VAR(Real, WindSpeed, 0, "");

        DEF_VAR(Real, Choppiness, 0, "");

        DEF_VAR(Real, GlobalShift, 0, "");


        DEF_VAR(Real, WindDirection, Real(60), "Wind direction");

        DEF_VAR(uint, Resolution, 512, "");

        DEF_VAR(Real, PatchSize, Real(512), "Real patch size");

        DEF_VAR(Real, TimeScale, Real(1), "");

    public:
        DEF_INSTANCE_STATE(HeightField<TDataType>, HeightField, "Height field");

    protected:
        void resetStates() override;

        void updateStates() override;
        void postUpdateStates() override;

    private:
        void resetWindType();

        std::vector<WindParam> mParams;  //A set of pre-defined configurations

        DArray2D<Complex> mH0;  //Initial spectrum
        DArray2D<Complex> mHt;  //Current spectrum

        DArray2D<Complex> mDxt;  //x-axis offset
        DArray2D<Complex> mDzt;  //z-axis offset

        DArray2D<Coord> mDisp;   //xyz-axis offset

        const Real g = 9.81f;          //Gravity

        Real mDirDepend = 0.07f;

        cufftHandle fftPlan;

        int mSpectrumWidth;  //with of spectrum
        int mSpectrumHeight;  //height of spectrum
    };

    IMPLEMENT_TCLASS(OceanPatch, TDataType)
}