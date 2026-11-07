"""Sequential local mapping from LocalMapping.cc and Optimizer.cc.

Each keyframe executes observation insertion, recent-point culling,
triangulation, two-way neighbor fusion, local BA, inertial initialization,
keyframe culling and inertial refinement in that order. The caller queues
loop recognition only after this boundary. No mapping worker thread is used.
Derived from ORB-SLAM3, University of Zaragoza contributors; GPL-3.0-or-later.
"""
import numpy as np
from .geometry import inverse_pose, make_pose, triangulate, transform_points
from .input import Sensor
from .matching import search_for_triangulation, fuse
from .optimization import ReprojectionObservation, local_bundle_adjustment


def reprojection_observations(keyframes, point_filter=None):
    """Build visible mono/stereo/body factors, with optional landmark-ID set."""
    observations = []
    for keyframe in keyframes:
        if getattr(keyframe,"bad",False):
            continue
        for index,point in enumerate(keyframe.map_points):
            if point is None or point.bad or (point_filter is not None and point.id not in point_filter):
                continue
            # Keyframe observations are authoritative after fusion/culling.
            if hasattr(keyframe,"map") and index not in point.observations.get(keyframe.id,(-1,-1)):
                continue
            camera = keyframe.camera_for(index)
            extrinsic = keyframe.T_right_left if keyframe.n_left >= 0 and index >= keyframe.n_left else None
            measurement = keyframe.features.points[index]
            if keyframe.n_left < 0 and keyframe.right_u[index] >= 0:
                measurement = np.r_[measurement,keyframe.right_u[index]]
            observations.append(ReprojectionObservation(keyframe.id,point.id,measurement,camera,
                                 keyframe.inverse_variances[keyframe.features.octaves[index]],
                                 keyframe.baseline_fx,extrinsic))
    return observations


def _erase_outlier_observations(map_, observations, outliers):
    pairs = {(observation.pose_id,observation.point_id)
             for observation,outlier in zip(observations,outliers) if outlier}
    for keyframe_id,point_id in sorted(pairs):
        keyframe,point = map_.keyframes.get(keyframe_id),map_.points.get(point_id)
        if keyframe is None or point is None or point.bad:
            continue
        for index in point.observations.get(keyframe_id,(-1,-1)):
            if index >= 0 and keyframe.map_points[index] is point:
                keyframe.map_points[index] = None
        point.erase_observation(keyframe)
    return len(pairs)


def select_visual_local_window(current):
    map_ = current.map
    local = [current]+[keyframe for keyframe in current.best_covisible(len(current.ordered_covisibility))
                      if keyframe.map is map_ and keyframe.id != current.id]
    point_ids = {point.id for keyframe in local for point in keyframe.map_points
                 if point is not None and not point.bad and point.map is map_}
    local_ids = {keyframe.id for keyframe in local}
    fixed_ids = set()
    if map_.initial_keyframe_id in local_ids:
        fixed_ids.add(map_.initial_keyframe_id)
    for point_id in sorted(point_ids):
        for observer in map_.points[point_id].observations:
            keyframe = map_.keyframes.get(observer)
            if observer not in local_ids and keyframe is not None and not keyframe.bad and keyframe.map is map_:
                fixed_ids.add(observer)
    included = local_ids | fixed_ids
    return local_ids,fixed_ids,point_ids,included


class LocalMapper:
    def __init__(self, atlas, sensor, vocabulary, database=None):
        self.atlas, self.sensor = atlas,Sensor(sensor)
        self.vocabulary, self.database = vocabulary,database
        self.recent_points = []
        self.current = None
        self.accepts_keyframes = True
        # A sequential caller normally leaves this zero. Setting it preserves
        # the source early exits when another queued keyframe is already known.
        self.pending_keyframes = 0
        self.abort_ba = False
        self.far_points, self.far_threshold = False,np.inf
        self.matches_inliers = 0
        self.inertial = None
        if self.sensor.inertial:
            from .inertial_mapping import InertialMapper
            self.inertial = InertialMapper(self.sensor.monocular)

    def reset(self):
        self.recent_points.clear()
        self.current = None
        self.pending_keyframes = 0
        self.abort_ba = False
        self.matches_inliers = 0
        self.accepts_keyframes = True
        if self.inertial is not None:
            from .inertial_mapping import InertialMapper
            self.inertial = InertialMapper(self.sensor.monocular)

    def process(self, keyframe, tracking_state=None):
        self.current, self.accepts_keyframes = keyframe,False
        diagnostics = {"new_points":0,"culled_points":0,"culled_keyframes":0,
                       "fusions":0,"local_ba":None,"inertial_reset_required":False}
        try:
            self.process_new_keyframe(keyframe)
            diagnostics["culled_points"] = self.cull_recent_points(keyframe)
            diagnostics["new_points"] = self.create_new_points(keyframe,tracking_state)
            self.abort_ba = False
            if not self.pending_keyframes:
                diagnostics["fusions"] = self.search_in_neighbors(keyframe)
                if len(keyframe.map.keyframes) > 2:
                    if self.sensor.inertial and keyframe.map.imu_initialized:
                        inertial_result = self.inertial.before_local_ba(keyframe)
                        diagnostics["inertial_reset_required"] = inertial_result.reset_required
                        diagnostics["local_ba"] = self.local_inertial_ba(keyframe)
                    else:
                        diagnostics["local_ba"] = self.local_visual_ba(keyframe)
                if self.inertial is not None and not keyframe.map.imu_initialized:
                    diagnostics["inertial_initialization"] = self.inertial.initialize(keyframe)
                diagnostics["culled_keyframes"] = self.cull_keyframes(keyframe)
                if self.inertial is not None:
                    state_name = getattr(tracking_state,"name",str(tracking_state))
                    diagnostics["inertial_refinement"] = self.inertial.refine(
                        keyframe,tracking_ok=tracking_state is None or state_name == "OK")
            self._sync_database_neighbors(keyframe.map)
            return diagnostics
        finally:
            self.accepts_keyframes = True

    def process_new_keyframe(self, keyframe):
        keyframe.compute_bow(self.vocabulary)
        for index,point in enumerate(keyframe.map_points):
            if point is None or point.bad:
                continue
            if keyframe.id not in point.observations:
                point.add_observation(keyframe,index)
                point.update_normal_and_depth()
                point.compute_descriptor()
            else:
                # This includes new depth points inserted by the tracker.
                self.recent_points.append(point)
        keyframe.update_connections()

    def cull_recent_points(self, current):
        threshold = 2 if self.sensor.monocular else 3
        retained,culled = [],0
        for point in self.recent_points:
            age = current.id-point.first_keyframe_id
            if point.bad:
                continue
            if point.found_ratio < .25 or (age >= 2 and point.observation_count <= threshold):
                point.set_bad()
                culled += 1
            elif age < 3:
                retained.append(point)
        self.recent_points = retained
        return culled

    def create_new_points(self, current, tracking_state=None):
        neighbor_count = 30 if self.sensor.monocular else 10
        neighbors = current.best_covisible(neighbor_count)
        if self.sensor.inertial:
            temporal,count = current,0
            while len(neighbors) <= neighbor_count and temporal.previous_id is not None and count < neighbor_count:
                temporal = current.map.keyframes.get(temporal.previous_id)
                if temporal is None:
                    break
                if temporal not in neighbors:
                    neighbors.append(temporal)
                count += 1
        factor = 1.5*(current.scales[1]/current.scales[0] if len(current.scales)>1 else 1.)
        recently_lost = getattr(tracking_state,"name",str(tracking_state)) == "RECENTLY_LOST"
        coarse = self.sensor.inertial and recently_lost and current.map.inertial_ba2
        created = 0
        for neighbor_index,neighbor in enumerate(neighbors):
            if neighbor_index > 0 and self.pending_keyframes:
                break
            baseline = np.linalg.norm(neighbor.center-current.center)
            if self.sensor.monocular:
                median_depth = neighbor.median_depth()
                if median_depth <= 0 or baseline/median_depth < .01:
                    continue
            elif baseline < neighbor.baseline:
                continue
            pairs = search_for_triangulation(current,neighbor,coarse=coarse)
            for i,j in pairs:
                pose1,pose2 = current.pose_for(i),neighbor.pose_for(j)
                camera1,camera2 = current.camera_for(i),neighbor.camera_for(j)
                center1,center2 = current.center_for(i),neighbor.center_for(j)
                ray1,ray2 = camera1.unproject(current.features.points[i]),camera2.unproject(neighbor.features.points[j])
                world_ray1,world_ray2 = pose1[:3,:3].T @ ray1,pose2[:3,:3].T @ ray2
                cosine = world_ray1 @ world_ray2/(np.linalg.norm(world_ray1)*np.linalg.norm(world_ray2))
                stereo1 = current.right_camera is None and current.right_u[i] >= 0
                stereo2 = neighbor.right_camera is None and neighbor.right_u[j] >= 0
                stereo_cosine1 = stereo_cosine2 = cosine+1
                if stereo1:
                    stereo_cosine1 = np.cos(2*np.arctan2(current.baseline/2,current.depth[i]))
                elif stereo2:  # the source deliberately uses else-if
                    stereo_cosine2 = np.cos(2*np.arctan2(neighbor.baseline/2,neighbor.depth[j]))
                if (cosine < min(stereo_cosine1,stereo_cosine2) and cosine > 0
                    and (stereo1 or stereo2 or cosine < (.9996 if self.sensor.inertial else .9998))):
                    position = triangulate(ray1,ray2,pose1[:3],pose2[:3])
                elif stereo1 and stereo_cosine1 < stereo_cosine2:
                    position = current.unproject_depth(i)
                elif stereo2 and stereo_cosine2 < stereo_cosine1:
                    position = neighbor.unproject_depth(j)
                else:
                    continue
                if position is None:
                    continue
                accepted = True
                for keyframe,index,pose,camera,stereo in ((current,i,pose1,camera1,stereo1),(neighbor,j,pose2,camera2,stereo2)):
                    local = transform_points(pose,position)
                    if local[2] <= 0:
                        accepted = False
                        break
                    pixel = camera.project(local)
                    residual = pixel-keyframe.features.points[index]
                    error = residual @ residual
                    cutoff = 5.991
                    if stereo:
                        # The active source also uses CURRENT bf for neighbor's
                        # right-image residual; retain that calibration choice.
                        error += (pixel[0]-current.baseline_fx/local[2]-keyframe.right_u[index])**2
                        cutoff = 7.8
                    if error*keyframe.inverse_variances[keyframe.features.octaves[index]] > cutoff:
                        accepted = False
                        break
                if not accepted:
                    continue
                distance1,distance2 = np.linalg.norm(position-center1),np.linalg.norm(position-center2)
                if distance1 == 0 or distance2 == 0:
                    continue
                if self.far_points and (distance1 >= self.far_threshold or distance2 >= self.far_threshold):
                    continue
                ratio = distance2/distance1
                octave_ratio = current.scales[current.features.octaves[i]]/neighbor.scales[neighbor.features.octaves[j]]
                if ratio*factor < octave_ratio or ratio > octave_ratio*factor:
                    continue
                point = self.atlas.add_point(position,current)
                point.add_observation(current,i)
                point.add_observation(neighbor,j)
                current.map_points[i],neighbor.map_points[j] = point,point
                point.compute_descriptor()
                point.update_normal_and_depth()
                self.recent_points.append(point)
                created += 1
        return created

    def search_in_neighbors(self, current):
        targets = current.best_covisible(30 if self.sensor.monocular else 10)
        marked = {keyframe.id for keyframe in targets}
        for target in targets.copy():
            for neighbor in target.best_covisible(20):
                if neighbor.id != current.id and neighbor.id not in marked:
                    targets.append(neighbor)
                    marked.add(neighbor.id)
            if self.abort_ba:
                break
        if self.sensor.inertial:
            previous = current.previous_id
            while len(targets) < 20 and previous is not None:
                neighbor = current.map.keyframes.get(previous)
                if neighbor is None:
                    break
                previous = neighbor.previous_id
                if not neighbor.bad and neighbor.id not in marked:
                    targets.append(neighbor)
                    marked.add(neighbor.id)
        fused = 0
        current_points = current.map_points.copy()
        for neighbor in targets:
            fused += fuse(neighbor,current_points)
            if neighbor.n_left >= 0:
                # Source Fuse(kf,points,true) binds true to float th, not
                # bRight: its second pass is LEFT with radius factor 1.
                fused += fuse(neighbor,current_points,threshold=1.)
        if self.abort_ba:
            return fused
        candidates,seen = [],set()
        for neighbor in targets:
            for point in neighbor.map_points:
                if point is not None and not point.bad and point.id not in seen:
                    seen.add(point.id)
                    candidates.append(point)
        fused += fuse(current,candidates)
        if current.n_left >= 0:
            fused += fuse(current,candidates,threshold=1.)
        for point in current.map_points:
            if point is not None and not point.bad:
                point.compute_descriptor()
                point.update_normal_and_depth()
        current.update_connections()
        return fused

    def local_visual_ba(self, current):
        local_ids,fixed_ids,point_ids,included = select_visual_local_window(current)
        if not fixed_ids:
            return {"skipped":"no fixed keyframe","optimized_keyframes":len(local_ids),"points":len(point_ids)}
        keyframes = [current.map.keyframes[i] for i in sorted(included)]
        observations = reprojection_observations(keyframes,point_ids)
        result = local_bundle_adjustment({keyframe.id:keyframe.pose for keyframe in keyframes},
                                         {i:current.map.points[i].position for i in sorted(point_ids)},
                                         observations,fixed_ids,inertial_map=current.map.inertial)
        erased = _erase_outlier_observations(current.map,observations,result.outliers)
        for identifier,pose in result.poses.items():
            keyframe = current.map.keyframes[identifier]
            keyframe.pose = pose
            if keyframe.body_state is not None and keyframe.imu_calibration is not None:
                # C++ GetImuRotation/GetImuPosition derive from the updated
                # camera pose. Keep the Python cached body pose coherent too.
                from .inertial_mapping import body_state_from_camera
                state = keyframe.body_state
                keyframe.body_state = body_state_from_camera(pose,keyframe.imu_calibration,
                                                             state.velocity,state.bias)
        for identifier,position in result.points.items():
            point = current.map.points.get(identifier)
            if point is not None and not point.bad:
                point.position = position
                point.update_normal_and_depth()
        current.map.change_index += 1
        return {"optimized_keyframes":len(local_ids),"fixed_keyframes":len(fixed_ids),
                "points":len(point_ids),"observations":len(observations),"erased_observations":erased,
                "initial_cost":result.initial_cost,"final_cost":result.final_cost}

    def local_inertial_ba(self, current):
        from .inertial_optimization import (InertialLink,select_inertial_local_window,
                                            local_inertial_bundle_adjustment)
        map_ = current.map
        large = self.matches_inliers > (75 if self.sensor.monocular else 100)
        predecessors = {identifier:keyframe.previous_id for identifier,keyframe in map_.keyframes.items()}
        observer_ids = {identifier:tuple(point.observations) for identifier,point in map_.points.items() if not point.bad}
        temporal,fixed,point_ids = select_inertial_local_window(current.id,predecessors,observer_ids,large,
                                                              bad_ids=[i for i,k in map_.keyframes.items() if k.bad])
        included = set(temporal) | set(fixed)
        keyframes = [map_.keyframes[i] for i in sorted(included)]
        if any(keyframe.body_state is None or keyframe.imu_calibration is None for keyframe in keyframes):
            raise ValueError("initialized inertial map requires body states and IMU calibration on every local keyframe")
        links = [InertialLink(map_.keyframes[i].previous_id,i,map_.keyframes[i].preintegrated)
                 for i in temporal if map_.keyframes[i].previous_id in included and map_.keyframes[i].preintegrated is not None]
        observations = reprojection_observations(keyframes,set(point_ids))
        result = local_inertial_bundle_adjustment({keyframe.id:keyframe.body_state for keyframe in keyframes},
                   {i:map_.points[i].position for i in point_ids},observations,links,
                   {keyframe.id:keyframe.imu_calibration for keyframe in keyframes},temporal,fixed,
                   large=large,reinitialize=not map_.inertial_ba2,
                   track_depths={i:getattr(map_.points[i],"track_depth",np.inf) for i in point_ids})
        if not result.accepted:
            return {"accepted":False,"initial_cost":result.initial_cost,"final_cost":result.final_cost}
        erased = _erase_outlier_observations(map_,observations,result.outliers)
        for identifier,state in result.states.items():
            keyframe = map_.keyframes[identifier]
            keyframe.body_state = state
            world_body = make_pose(state.rotation,state.position)
            keyframe.pose = inverse_pose(world_body @ keyframe.imu_calibration.T_body_camera)
            if keyframe.preintegrated is not None:
                keyframe.preintegrated.set_new_bias(state.bias)
        for identifier,position in result.points.items():
            point = map_.points.get(identifier)
            if point is not None and not point.bad:
                point.position = position
                point.update_normal_and_depth()
        map_.change_index += 1
        return {"accepted":True,"optimized_keyframes":len(temporal),"fixed_keyframes":len(fixed),
                "points":len(point_ids),"observations":len(observations),"erased_observations":erased,
                "initial_cost":result.initial_cost,"final_cost":result.final_cost}

    def cull_keyframes(self, current):
        map_ = current.map
        threshold = .5 if self.sensor.inertial and not self.sensor.monocular else .9
        last = current
        for _ in range(21):
            if last.previous_id is None or last.previous_id not in map_.keyframes:
                break
            last = map_.keyframes[last.previous_id]
        # UpdateBestCovisibles includes all connections, including weak ones.
        ordered = sorted(current.covisibility,key=lambda i:(current.covisibility[i],i),reverse=True)
        current.ordered_covisibility = ordered.copy()
        culled = 0
        for count,identifier in enumerate(ordered,start=1):
            keyframe = map_.keyframes.get(identifier)
            if keyframe is None or keyframe.bad or identifier == map_.initial_keyframe_id:
                continue
            considered,redundant = 0,0
            for index,point in enumerate(keyframe.map_points):
                if point is None or point.bad:
                    continue
                if not self.sensor.monocular and (keyframe.depth[index] > keyframe.depth_threshold or keyframe.depth[index] < 0):
                    continue
                considered += 1
                if point.observation_count <= 3:
                    continue
                level = keyframe.features.octaves[index]
                other_observers = 0
                for observer,indices in sorted(point.observations.items()):
                    if observer == identifier:
                        continue
                    other = map_.keyframes.get(observer)
                    if other is None:
                        continue
                    other_levels = [other.features.octaves[i] for i in indices if i >= 0]
                    if other_levels and min(other_levels) <= level+1:
                        other_observers += 1
                        if other_observers > 3:
                            break
                redundant += other_observers > 3  # retain source's strict >3
            if redundant > threshold*considered:
                remove = not self.sensor.inertial
                if self.sensor.inertial and len(map_.keyframes)>21 and identifier <= current.id-2:
                    previous,next_ = map_.keyframes.get(keyframe.previous_id),map_.keyframes.get(keyframe.next_id)
                    if previous is not None and next_ is not None:
                        interval = next_.timestamp-previous.timestamp
                        remove = ((map_.imu_initialized and identifier < last.id and interval<3) or interval<.5)
                        if not remove and not map_.inertial_ba2 and interval<3:
                            if keyframe.body_state is not None and previous.body_state is not None:
                                remove = np.linalg.norm(keyframe.body_state.position-previous.body_state.position)<.02
                        if remove:
                            if next_.preintegrated is None or keyframe.preintegrated is None:
                                raise ValueError("temporal keyframe culling requires both IMU preintegrations")
                            next_.preintegrated.merge_previous(keyframe.preintegrated)
                            next_.previous_id,previous.next_id = previous.id,next_.id
                            keyframe.previous_id = keyframe.next_id = None
                if remove:
                    keyframe.set_bad(self.database)
                    culled += keyframe.bad
            if (count>20 and self.abort_ba) or count>100:
                break
        return culled

    def _sync_database_neighbors(self, map_):
        if self.database is None:
            return
        for identifier,keyframe in map_.keyframes.items():
            if identifier in self.database.entries:
                self.database.entries[identifier].neighbors = keyframe.ordered_covisibility.copy()
