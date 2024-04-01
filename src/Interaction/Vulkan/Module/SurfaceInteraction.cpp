#include "SurfaceInteraction.h"
#include <iostream>
#include <OrbitCamera.h>

namespace dyno
{
    template <typename TDataType>
    SurfaceInteraction<TDataType>::SurfaceInteraction() {
        this->ray1 = TRay3D<Real>();
        this->ray2 = TRay3D<Real>();
        this->isPressed = false;

        this->outOtherTriangleSet()->setDataPtr(std::make_shared<TriangleSet3f>());
        this->outOtherTriangleSet()->getDataPtr()->getTriangles().resize(0);
        this->outSelectedTriangleSet()->setDataPtr(std::make_shared<TriangleSet3f>());
        this->outSelectedTriangleSet()->getDataPtr()->getTriangles().resize(0);

        this->addKernel("SurfaceRay",
                        std::make_shared<VkProgram>(BUFFER(Coord), BUFFER(int), BUFFER(int), BUFFER(Real),
                                                    BUFFER(Triangle), CONSTANT(uint), CONSTANT(TRay3D<Real>)));
        this->kernel("SurfaceRay")
            ->load(getSpvFile("shaders/glsl/interaction/SurfaceRay.comp.spv"));

        this->addKernel("Nearest",
                        std::make_shared<VkProgram>(BUFFER(int), BUFFER(int), CONSTANT(uint), CONSTANT(int)));
        this->kernel("Nearest")->load(VkSystem::instance()->getAssetPath() /
                                      "shaders/glsl/interaction/Nearest.comp.spv");

        this->addKernel("NeighborTrisDiffuse",
                        std::make_shared<VkProgram>(BUFFER(Triangle), BUFFER(Coord), BUFFER(int), BUFFER(int),
                                                    CONSTANT(uint), CONSTANT(Real)));
        this->kernel("NeighborTrisDiffuse")
            ->load(getSpvFile("shaders/glsl/interaction/NeighborTrisDiffuse.comp.spv"));

        this->addKernel("TriVisibleFilter",
                        std::make_shared<VkProgram>(BUFFER(Triangle), BUFFER(Coord), BUFFER(int), BUFFER(int),
                                                    CONSTANT(uint), CONSTANT(TRay3D<Real>)));
        this->kernel("TriVisibleFilter")
            ->load(getSpvFile("shaders/glsl/interaction/TriVisibleFilter.comp.spv"));

        this->addKernel("Tri2Quad", std::make_shared<VkProgram>(BUFFER(int), BUFFER(int), CONSTANT(uint)));
        this->kernel("Tri2Quad")
            ->load(getSpvFile("shaders/glsl/interaction/Tri2Quad.comp.spv"));

        this->addKernel("SurfaceAssignOut",
                        std::make_shared<VkProgram>(BUFFER(Triangle), BUFFER(Triangle), BUFFER(Triangle), BUFFER(int),
                                                    BUFFER(int), BUFFER(int), BUFFER(int), CONSTANT(uint)));
        this->kernel("SurfaceAssignOut")
            ->load(getSpvFile("shaders/glsl/interaction/SurfaceAssignOut.comp.spv"));

        this->addKernel("Surface2Point",
                        std::make_shared<VkProgram>(BUFFER(Triangle), BUFFER(int), BUFFER(int), CONSTANT(uint)));
        this->kernel("Surface2Point")
            ->load(getSpvFile("shaders/glsl/interaction/Surface2Point.comp.spv"));

        this->addKernel("S2IndexOut",
                        std::make_shared<VkProgram>(BUFFER(int), BUFFER(int), BUFFER(int), CONSTANT(uint)));
        this->kernel("S2IndexOut")
            ->load(getSpvFile("shaders/glsl/interaction/S2IndexOut.comp.spv"));

        this->addKernel("QuadOutput", std::make_shared<VkProgram>(BUFFER(int), BUFFER(int), CONSTANT(uint)));
        this->kernel("QuadOutput")
            ->load(getSpvFile("shaders/glsl/interaction/QuadOutput.comp.spv"));

        this->addKernel("QuadIndexOutput", std::make_shared<VkProgram>(BUFFER(int), BUFFER(int), CONSTANT(uint)));
        this->kernel("QuadIndexOutput")
            ->load(getSpvFile("shaders/glsl/interaction/QuadIndexOutput.comp.spv"));


        this->addKernel("SurfaceBox", std::make_shared<VkProgram>(
                                       BUFFER(Coord), BUFFER(Triangle), BUFFER(int), BUFFER(int), CONSTANT(uint),
                                       CONSTANT(TPlane3D<Real>), CONSTANT(TPlane3D<Real>), CONSTANT(TPlane3D<Real>),
                                       CONSTANT(TPlane3D<Real>), CONSTANT(TRay3D<Real>)));
        this->kernel("SurfaceBox")->load(VkSystem::instance()->getAssetPath() /
                                      "shaders/glsl/interaction/SurfaceBox.comp.spv");
        this->addKernel("Merge", std::make_shared<VkProgram>(BUFFER(int), BUFFER(int), BUFFER(int), BUFFER(int),
            CONSTANT(uint), CONSTANT(int)));
        this->kernel("Merge")
            ->load(getSpvFile("shaders/glsl/interaction/Merge.comp.spv"));
    }

    template <typename TDataType>
    void SurfaceInteraction<TDataType>::onEvent(PMouseEvent event) {
        if (!event.altKeyPressed()) {
            if (camera == nullptr) {
                this->camera = event.camera;
            }
            this->varToggleMultiSelect()->setValue(false);
            if (event.shiftKeyPressed() || event.controlKeyPressed()) {
                this->varToggleMultiSelect()->setValue(true);
                if (event.shiftKeyPressed() && !event.controlKeyPressed()) {
                    this->varMultiSelectionType()->getDataPtr()->setCurrentKey(0);
                }
                else if (!event.shiftKeyPressed() && event.controlKeyPressed()) {
                    this->varMultiSelectionType()->getDataPtr()->setCurrentKey(1);
                }
                else if (event.shiftKeyPressed() && event.controlKeyPressed()) {
                    this->varMultiSelectionType()->getDataPtr()->setCurrentKey(2);
                    ;
                }
            }
            if (event.actionType == AT_PRESS) {
                this->camera = event.camera;
                this->isPressed = true;
                // printf("Mouse pressed: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x,
                // event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y,
                // event.ray.direction.z);
                this->ray1.origin = event.ray.origin;
                this->ray1.direction = event.ray.direction;
                this->x1 = event.x;
                this->y1 = event.y;
                if (this->varSurfacePickingType()->getValue() == PickingTypeSelection::Both ||
                    this->varSurfacePickingType()->getValue() == PickingTypeSelection::Click)
                {
                    this->calcIntersectClick();
                    this->printInfoClick();
                }
            }
            else if (event.actionType == AT_RELEASE) {
                this->isPressed = false;
                // printf("Mouse released: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x,
                // event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y,
                // event.ray.direction.z);
                this->ray2.origin = event.ray.origin;
                this->ray2.direction = event.ray.direction;
                this->x2 = event.x;
                this->y2 = event.y;
                if (this->varToggleMultiSelect()->getValue() && this->varTogglePicker()->getValue()) {
                    this->mergeIndex();
                    this->printInfoDragRelease();
                }
            }
            else {
                // printf("Mouse repeated: Origin: %f %f %f; Direction: %f %f %f \n", event.ray.origin.x,
                // event.ray.origin.y, event.ray.origin.z, event.ray.direction.x, event.ray.direction.y,
                // event.ray.direction.z);
                if (this->isPressed) {
                    this->ray2.origin = event.ray.origin;
                    this->ray2.direction = event.ray.direction;
                    this->x2 = event.x;
                    this->y2 = event.y;
                    if (abs(this->x2 - this->x1) <= 3 && abs(this->y2 - this->y1) <= 3) {
                        if (this->varSurfacePickingType()->getValue() == PickingTypeSelection::Both ||
                            this->varSurfacePickingType()->getValue() == PickingTypeSelection::Click)
                            this->calcIntersectClick();
                    }
                    else {
                        if (this->varSurfacePickingType()->getValue() == PickingTypeSelection::Both ||
                            this->varSurfacePickingType()->getValue() == PickingTypeSelection::Drag)
                        {
                            this->calcIntersectDrag();
                            this->printInfoDragging();
                        }
                    }
                }
            }
        }
    }

    template <typename TDataType>
    void SurfaceInteraction<TDataType>::calcSurfaceIntersectClick() {
        VkCompContext::Holder holder;
        holder.delaySubmit(true);

        TriangleSet3f initialTriangleSet = this->inInitialTriangleSet()->getData();
        DArray<Coord> points = initialTriangleSet.getPoints();
        DArray<Triangle> triangles = initialTriangleSet.getTriangles();
        DArray<int> intersected(triangles.size());
        DArray<int> unintersected(triangles.size());
        this->tempNumT = triangles.size();

        DArray<Real> triDistance(triangles.size());

        VkConstant<uint> vk_num {triangles.size()};
        VkConstant<TRay3D<Real>> vk_ray {this->ray1};
        this->kernel("SurfaceRay")
            ->submit(vkDispatchSize(vk_num, 64), points.handle(), intersected.handle(), unintersected.handle(),
                    triDistance.handle(), triangles.handle(), &vk_num, &vk_ray);

        int min_index = this->mMin.reduce(*triDistance.handle());

        vk_num.setValue(intersected.size());
        VkConstant<int> vk_min_index {min_index};
        this->kernel("Nearest")->submit(vkDispatchSize(vk_num, 64), intersected.handle(), unintersected.handle(),
                                       &vk_num, &vk_min_index);
        if (this->varToggleFlood()->getValue()) {
            int intersected_size_t_o = 0;
            int intersected_size_t = 1;
            while (intersected_size_t > intersected_size_t_o && intersected_size_t < triangles.size()) {
                intersected_size_t_o = intersected_size_t;
                vk_num.setValue(triangles.size());
                VkConstant<Real> vk_diff {Real(this->varFloodAngle()->getValue() / 180.0f * M_PI)};
                this->kernel("NeighborTrisDiffuse")
                    ->submit(vkDispatchSize2D(vk_num, vk_num, 8), triangles.handle(), points.handle(),
                            intersected.handle(), unintersected.handle(), &vk_num, &vk_diff);
                intersected_size_t = this->mReduce.reduce(*intersected.handle());
            }
        }

        if (this->varToggleVisibleFilter()->getValue()) {
            vk_num.setValue(triangles.size());
            this->kernel("TriVisibleFilter")
                ->submit(vkDispatchSize(vk_num, 64), triangles.handle(), points.handle(), intersected.handle(),
                        unintersected.handle(), &vk_num, &vk_ray);
        }

        if (this->varToggleQuad()->getValue()) {
            vk_num.setValue(triangles.size());
            this->kernel("Tri2Quad")
                ->submit(vkDispatchSize(vk_num, 64), intersected.handle(), unintersected.handle(), &vk_num);
        }

        this->tempTriIntersectedIndex.assign(intersected);

		DArray<int> outIntersected(intersected.size());
		DArray<int> outUnintersected(unintersected.size());

        if (this->varToggleMultiSelect()->getData()) {
            if (this->triIntersectedIndex.size() == 0) {
                this->triIntersectedIndex.resize(triangles.size());
            }
			outIntersected.resize(intersected.size());
			outUnintersected.resize(unintersected.size());

            vk_num.setValue(this->triIntersectedIndex.size());
            VkConstant<int> vk_select_type {(int)this->varMultiSelectionType()->getValue().currentKey()};
            this->kernel("Merge")->submit(vkDispatchSize(vk_num, 64), this->triIntersectedIndex.handle(),
                                         intersected.handle(), outIntersected.handle(), outUnintersected.handle(),
                                         &vk_num, &vk_select_type);

            intersected.assign(outIntersected);
            unintersected.assign(outUnintersected);
        }
        else {
            this->triIntersectedIndex.assign(intersected);
        }

        DArray<int> intersected_o;
        intersected_o.assign(intersected);

        int intersected_size = mReduce.reduce(*intersected.handle());
        DArray<int> outTriangleIndex;
        outTriangleIndex.resize(intersected_size);
        mScan.scan(*intersected.handle(), *intersected.handle(), VkScan<int>::Exclusive);
        DArray<Triangle> intersected_triangles;
        intersected_triangles.resize(intersected_size);

        int unintersected_size = mReduce.reduce(*unintersected.handle());
        mScan.scan(*unintersected.handle(), *unintersected.handle(), VkScan<int>::Exclusive);
        DArray<Triangle> unintersected_triangles;
        unintersected_triangles.resize(unintersected_size);

        vk_num.setValue(triangles.size());
        this->kernel("SurfaceAssignOut")
            ->submit(vkDispatchSize(vk_num, 64), triangles.handle(), intersected_triangles.handle(),
                    unintersected_triangles.handle(), outTriangleIndex.handle(), intersected.handle(),
                    unintersected.handle(), intersected_o.handle(), &vk_num);

        DArray<int> s2PSelected;
        s2PSelected.resize(points.size());
        vk_num.setValue(outTriangleIndex.size());
        this->kernel("Surface2Point")
            ->submit(vkDispatchSize(vk_num, 64), triangles.handle(), outTriangleIndex.handle(), s2PSelected.handle(),
                    &vk_num);

        int s2PSelectedSize = mReduce.reduce(*s2PSelected.handle());
        DArray<int> s2PSelected_o;
        s2PSelected_o.assign(s2PSelected);
        mScan.scan(*s2PSelected.handle(), *s2PSelected.handle(), VkScan<int>::Exclusive);

        DArray<int> s2PSelectedIndex;
        s2PSelectedIndex.resize(s2PSelectedSize);
        vk_num.setValue(s2PSelected.size());
        this->kernel("S2IndexOut")
            ->submit(vkDispatchSize(vk_num, 64), s2PSelected.handle(), s2PSelected_o.handle(), s2PSelectedIndex.handle(),
                    &vk_num);

		DArray<int> intersected_q;
		DArray<int> outQuadIndex;
        if (this->varToggleQuad()->getValue()) {
            intersected_q.resize(triangles.size() / 2);
            vk_num.setValue(triangles.size());
            this->kernel("QuadOutput")
                ->submit(vkDispatchSize(vk_num, 64), intersected_o.handle(), intersected_q.handle(), &vk_num);

            intersected_o.assign(intersected_q);

            outQuadIndex.resize(outTriangleIndex.size() / 2);
            vk_num.setValue(outQuadIndex.size());
            this->kernel("QuadIndexOutput")
                ->submit(vkDispatchSize(vk_num, 64), outTriangleIndex.handle(), outQuadIndex.handle(), &vk_num);
            outTriangleIndex.assign(outQuadIndex);
        }

        this->tempNumS = intersected_size;
        this->outSelectedTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
        this->outSelectedTriangleSet()->getDataPtr()->setTriangles(intersected_triangles);
        this->outOtherTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
        this->outOtherTriangleSet()->getDataPtr()->setTriangles(unintersected_triangles);
        if (this->varToggleIndexOutput()->getValue()) {
            this->outTriangleIndex()->getDataPtr()->assign(outTriangleIndex);
            this->outSur2PointIndex()->getDataPtr()->assign(s2PSelectedIndex);
        }
        else {
            this->outTriangleIndex()->getDataPtr()->assign(intersected_o);
            this->outSur2PointIndex()->getDataPtr()->assign(s2PSelected_o);
        }
    }

    template <typename TDataType>
    void SurfaceInteraction<TDataType>::calcSurfaceIntersectDrag() {
        VkCompContext::Holder holder;
        holder.delaySubmit(true);

        if (x1 == x2) {
            x2 += 1.0f;
        }
        if (y1 == y2) {
            y2 += 1.0f;
        }
        TRay3D<Real> ray1 = this->camera->castRayInWorldSpace((float)x1, (float)y1);
        TRay3D<Real> ray2 = this->camera->castRayInWorldSpace((float)x2, (float)y2);
        TRay3D<Real> ray3 = this->camera->castRayInWorldSpace((float)x1, (float)y2);
        TRay3D<Real> ray4 = this->camera->castRayInWorldSpace((float)x2, (float)y1);

        VkConstant<TPlane3D<Real>> plane13 {TPlane3D<Real>(ray1.origin, ray1.direction.cross(ray3.direction))};
        VkConstant<TPlane3D<Real>> plane42 {TPlane3D<Real>(ray2.origin, ray2.direction.cross(ray4.direction))};
        VkConstant<TPlane3D<Real>> plane14 {TPlane3D<Real>(ray4.origin, ray1.direction.cross(ray4.direction))};
        VkConstant<TPlane3D<Real>> plane32 {TPlane3D<Real>(ray3.origin, ray2.direction.cross(ray3.direction))};

        TriangleSet3f initialTriangleSet = this->inInitialTriangleSet()->getData();
        DArray<Coord> points = initialTriangleSet.getPoints();
        DArray<Triangle> triangles = initialTriangleSet.getTriangles();
        DArray<int> intersected(triangles.size());
        DArray<int> unintersected(triangles.size());
        this->tempNumT = triangles.size();

        VkConstant<uint> vk_num {triangles.size()};
        VkConstant<TRay3D<Real>> vk_ray {this->ray2};
        this->kernel("SurfaceBox")
            ->submit(vkDispatchSize(vk_num, 64), points.handle(), triangles.handle(), intersected.handle(),
                    unintersected.handle(), &vk_num, &plane13, &plane42, &plane14, &plane32, &vk_ray);

        vk_ray.setValue(this->ray1);

        if (this->varToggleVisibleFilter()->getValue()) {
            vk_num.setValue(triangles.size());
            this->kernel("TriVisibleFilter")
                ->submit(vkDispatchSize(vk_num, 64), triangles.handle(), points.handle(), intersected.handle(),
                        unintersected.handle(), &vk_num, &vk_ray);
        }

        if (this->varToggleQuad()->getValue()) {
            vk_num.setValue(triangles.size());
            this->kernel("Tri2Quad")
                ->submit(vkDispatchSize(vk_num, 64), intersected.handle(), unintersected.handle(), &vk_num);
        }
        this->tempTriIntersectedIndex.assign(intersected);

        if (this->varToggleMultiSelect()->getData()) {
            if (this->triIntersectedIndex.size() == 0) {
                this->triIntersectedIndex.resize(triangles.size());
            }
            DArray<int> outIntersected(intersected.size());
            DArray<int> outUnintersected(unintersected.size());
            VkConstant<int> vk_select_type {(int)this->varMultiSelectionType()->getValue().currentKey()};
            this->kernel("Merge")->submit(vkDispatchSize(vk_num, 64), this->triIntersectedIndex.handle(),
                                         intersected.handle(), outIntersected.handle(), outUnintersected.handle(),
                                         &vk_num, &vk_select_type);

            intersected.assign(outIntersected);
            unintersected.assign(outUnintersected);
            outIntersected.clear();
            outUnintersected.clear();
        }
        else {
            this->triIntersectedIndex.assign(intersected);
        }

		DArray<int> intersected_o;
        intersected_o.assign(intersected);

        int intersected_size = mReduce.reduce(*intersected.handle());
        mScan.scan(*intersected.handle(), *intersected.handle(), VkScan<int>::Exclusive);

        int unintersected_size = mReduce.reduce(*unintersected.handle());
        mScan.scan(*unintersected.handle(), *unintersected.handle(), VkScan<int>::Exclusive);

        DArray<int> outTriangleIndex(intersected_size);
        DArray<Triangle> intersected_triangles(intersected_size);
        DArray<Triangle> unintersected_triangles(unintersected_size);

		vk_num.setValue(triangles.size());
        this->kernel("SurfaceAssignOut")
            ->submit(vkDispatchSize(vk_num, 64), triangles.handle(), intersected_triangles.handle(),
                    unintersected_triangles.handle(), outTriangleIndex.handle(), intersected.handle(),
                    unintersected.handle(), intersected_o.handle(), &vk_num);

        DArray<int> s2PSelected(points.size());

        vk_num.setValue(outTriangleIndex.size());
        this->kernel("Surface2Point")
            ->submit(vkDispatchSize(vk_num, 64), triangles.handle(), outTriangleIndex.handle(), s2PSelected.handle(),
                    &vk_num);

        int s2PSelectedSize = mReduce.reduce(*s2PSelected.handle());

        DArray<int> s2PSelected_o;
        s2PSelected_o.assign(s2PSelected);

        mScan.scan(*s2PSelected.handle(), *s2PSelected.handle(), VkScan<int>::Exclusive);

        DArray<int> s2PSelectedIndex(s2PSelectedSize);
        vk_num.setValue(s2PSelected.size());
        this->kernel("S2IndexOut")
            ->submit(vkDispatchSize(vk_num, 64), s2PSelected.handle(), s2PSelected_o.handle(), s2PSelectedIndex.handle(),
                    &vk_num);

        if (this->varToggleQuad()->getValue()) {
            DArray<int> intersected_q(triangles.size() / 2);
            vk_num.setValue(triangles.size());
            this->kernel("QuadOutput")
                ->submit(vkDispatchSize(vk_num, 64), intersected_o.handle(), intersected_q.handle(), &vk_num);

            intersected_o.assign(intersected_q);

            DArray<int> outQuadIndex(outTriangleIndex.size() / 2);
            vk_num.setValue(outQuadIndex.size());
            this->kernel("QuadIndexOutput")
                ->submit(vkDispatchSize(vk_num, 64), outTriangleIndex.handle(), outQuadIndex.handle(), &vk_num);
            outTriangleIndex.assign(outQuadIndex);
        }

        this->tempNumS = intersected_size;
        this->outSelectedTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
        this->outSelectedTriangleSet()->getDataPtr()->setTriangles(intersected_triangles);
        this->outOtherTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
        this->outOtherTriangleSet()->getDataPtr()->setTriangles(unintersected_triangles);
        if (this->varToggleIndexOutput()->getValue()) {
            this->outTriangleIndex()->getDataPtr()->assign(outTriangleIndex);
            this->outSur2PointIndex()->getDataPtr()->assign(s2PSelectedIndex);
        }
        else {
            this->outTriangleIndex()->getDataPtr()->assign(intersected_o);
            this->outSur2PointIndex()->getDataPtr()->assign(s2PSelected_o);
        }
    }

    template <typename TDataType>
    void SurfaceInteraction<TDataType>::mergeIndex() {
        VkCompContext::Holder holder;
        holder.delaySubmit(true);

        TriangleSet3f initialTriangleSet = this->inInitialTriangleSet()->getData();
        DArray<Coord> points = initialTriangleSet.getPoints();
        DArray<Triangle> triangles = initialTriangleSet.getTriangles();
        DArray<int> intersected(triangles.size());
        DArray<int> unintersected(triangles.size());
        this->tempNumT = triangles.size();

        DArray<int> outIntersected(intersected.size());
        DArray<int> outUnintersected(unintersected.size());

        VkConstant<uint> vk_num {this->triIntersectedIndex.size()};

        if (this->varToggleMultiSelect()->getData()) {
            VkConstant<int> vk_select_type {(int)this->varMultiSelectionType()->getValue().currentKey()};
            this->kernel("Merge")->submit(vkDispatchSize(vk_num, 64), this->triIntersectedIndex.handle(),
                                         this->tempTriIntersectedIndex.handle(), outIntersected.handle(), outUnintersected.handle(),
                                         &vk_num, &vk_select_type);

        }

        intersected.assign(outIntersected);
        unintersected.assign(outUnintersected);
        this->triIntersectedIndex.assign(intersected);

		DArray<int> intersected_o;
		intersected_o.assign(intersected);

		int intersected_size = mReduce.reduce(*intersected.handle());
		mScan.scan(*intersected.handle(), *intersected.handle(), VkScan<int>::Exclusive);

		int unintersected_size = mReduce.reduce(*unintersected.handle());
		mScan.scan(*unintersected.handle(), *unintersected.handle(), VkScan<int>::Exclusive);

		DArray<int> outTriangleIndex(intersected_size);
		DArray<Triangle> intersected_triangles(intersected_size);
		DArray<Triangle> unintersected_triangles(unintersected_size);

		vk_num.setValue(triangles.size());
		this->kernel("SurfaceAssignOut")
			->submit(vkDispatchSize(vk_num, 64), triangles.handle(), intersected_triangles.handle(),
				unintersected_triangles.handle(), outTriangleIndex.handle(), intersected.handle(),
				unintersected.handle(), intersected_o.handle(), &vk_num);

		DArray<int> s2PSelected(points.size());
		vk_num.setValue(outTriangleIndex.size());
		this->kernel("Surface2Point")
			->submit(vkDispatchSize(vk_num, 64), triangles.handle(), outTriangleIndex.handle(), s2PSelected.handle(),
				&vk_num);

		int s2PSelectedSize = mReduce.reduce(*s2PSelected.handle());

		DArray<int> s2PSelected_o;
		s2PSelected_o.assign(s2PSelected);

		mScan.scan(*s2PSelected.handle(), *s2PSelected.handle(), VkScan<int>::Exclusive);

		DArray<int> s2PSelectedIndex(s2PSelectedSize);
		vk_num.setValue(s2PSelected.size());
		this->kernel("S2IndexOut")
			->submit(vkDispatchSize(vk_num, 64), s2PSelected.handle(), s2PSelected_o.handle(), s2PSelectedIndex.handle(),
				&vk_num);

        this->tempNumS = intersected_size;
        this->outSelectedTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
        this->outSelectedTriangleSet()->getDataPtr()->setTriangles(intersected_triangles);
        this->outOtherTriangleSet()->getDataPtr()->copyFrom(initialTriangleSet);
        this->outOtherTriangleSet()->getDataPtr()->setTriangles(unintersected_triangles);
        if (this->varToggleIndexOutput()->getValue())
        {
            this->outTriangleIndex()->getDataPtr()->assign(outTriangleIndex);
            this->outSur2PointIndex()->getDataPtr()->assign(s2PSelectedIndex);
        }
        else
        {
            this->outTriangleIndex()->getDataPtr()->assign(intersected_o);
            this->outSur2PointIndex()->getDataPtr()->assign(s2PSelected_o);
        }
    }

    template <typename TDataType>
    void SurfaceInteraction<TDataType>::printInfoClick() {
        std::cout << "----------surface picking: click----------" << std::endl;
        std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
        std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
    }

    template <typename TDataType>
    void SurfaceInteraction<TDataType>::printInfoDragging() {
        std::cout << "----------surface picking: dragging----------" << std::endl;
        std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
        std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
    }

    template <typename TDataType>
    void SurfaceInteraction<TDataType>::printInfoDragRelease() {
        std::cout << "----------surface picking: drag release----------" << std::endl;
        std::cout << "multiple picking: " << this->varToggleMultiSelect()->getValue() << std::endl;
        std::cout << "selected num/ total num:" << this->tempNumS << "/" << this->tempNumT << std::endl;
    }

    template <typename TDataType>
    void SurfaceInteraction<TDataType>::calcIntersectClick() {
        if (this->varTogglePicker()->getData()) calcSurfaceIntersectClick();
    }

    template <typename TDataType>
    void SurfaceInteraction<TDataType>::calcIntersectDrag() {
        if (this->varTogglePicker()->getData()) calcSurfaceIntersectDrag();
    }

    DEFINE_CLASS(SurfaceInteraction);
} // namespace dyno