#pragma once
#include "Module/MouseInputModule.h"
#include "Module/TopologyModule.h"
#include "Topology/TriangleSet.h"
#include "Catalyzer/VkMin.h"

namespace dyno
{
    template <typename TDataType>
    class PointInteraction : public MouseInputModule {
        DECLARE_TCLASS(PointInteraction, TDataType)
    public:
        typedef typename TDataType::Real Real;
        typedef typename TDataType::Coord Coord;
        typedef typename TopologyModule::Edge Edge;
        typedef typename TopologyModule::Triangle Triangle;

		static_assert(sizeof(Coord) == 4*4);

        PointInteraction();
        virtual ~PointInteraction();

        void calcIntersectClick();
        void calcIntersectDrag();
        void calcPointIntersectClick();
        void calcPointIntersectDrag();

        void mergeIndex();

        void printInfoClick();
        void printInfoDragging();
        void printInfoDragRelease();

        DEF_INSTANCE_IN(PointSet3f, InitialPointSet, "");
        DEF_INSTANCE_OUT(PointSet3f, SelectedPointSet, "");
        DEF_INSTANCE_OUT(PointSet3f, OtherPointSet, "");
        DEF_ARRAY_OUT(int, PointIndex, DeviceType::GPU, "");

        DECLARE_ENUM(PickingTypeSelection, Click = 0, Drag = 1, Both = 2);

        DEF_ENUM(PickingTypeSelection, PointPickingType, PickingTypeSelection::Both, "");

        DECLARE_ENUM(MultiSelectionType, OR = 0, XOR = 1, C = 2);

        DEF_ENUM(MultiSelectionType, MultiSelectionType, MultiSelectionType::OR, "");

        DEF_VAR(Real, InteractionRadius, 0.01, "The radius of interaction");

        DEF_VAR(bool, TogglePicker, true, "The toggle of picker");

        DEF_VAR(bool, ToggleMultiSelect, false, "The toggle of multiple selection");

        DEF_VAR(bool, ToggleIndexOutput, true, "The toggle of index output");

    protected:
        void onEvent(PMouseEvent event) override;

    private:
        std::shared_ptr<Camera> camera;
        TRay3D<Real> ray1, ray2;
        Real x1, y1, x2, y2;
        bool isPressed;
        int tempNumT, tempNumS;

        VkMinElement<Real> mMin;
        VkReduce<int> mReduce;
        VkScan<int> mScan;

        DArray<int> pointIntersectedIndex;
        DArray<int> tempPointIntersectedIndex;
    };

    IMPLEMENT_TCLASS(PointInteraction, TDataType)
} // namespace dyno
