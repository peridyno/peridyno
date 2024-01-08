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
#include "Node/ParametricModel.h"

#include "Topology/HeightField.h"
#include "FilePath.h"
namespace dyno {

    template<typename TDataType>
    class LandScape : public ParametricModel<TDataType>
    {
        DECLARE_TCLASS(LandScape, TDataType)
    public:
        typedef typename TDataType::Real Real;
        typedef typename TDataType::Coord Coord;
 
        LandScape();
        ~LandScape();

    public:
 
        DEF_VAR(Real, PatchSize, Real(256), "Real patch size");

        DEF_VAR(FilePath, FileName, "", "");

        DEF_INSTANCE_STATE(HeightField<TDataType>, HeightField, "Topology");

    protected:
        void resetStates() override;

        void callbackTransform();

        void callbackLoadFile();

    private:
        std::string fileName;

        DArray2D<Real> mInitialHeights;
    };

    IMPLEMENT_TCLASS(LandScape, TDataType)
}