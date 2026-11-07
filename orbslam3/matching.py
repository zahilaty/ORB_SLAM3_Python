"""ORB-SLAM3 descriptor association, projection search and landmark fusion.

Translated from src/ORBmatcher.cc (University of Zaragoza contributors,
GPL-3.0-or-later). Thresholds, strict/non-strict ratio tests and the source's
unusual orientation binning are retained. Frame/KeyFrame records carry the
feature arrays; this module contains no image loading or display code.
"""
import math
import numpy as np
from .features import hamming
from .geometry import inverse_pose, transform_points

TH_LOW, TH_HIGH, HISTOGRAM_LENGTH = 50, 100, 30


def compute_three_maxima(histogram):
    maxima, indices = [0,0,0], [-1,-1,-1]
    for index, entries in enumerate(histogram):
        size = len(entries)
        for slot in range(3):
            if size > maxima[slot]:
                maxima.insert(slot,size)
                indices.insert(slot,index)
                maxima.pop()
                indices.pop()
                break
    if maxima[1] < 0.1*maxima[0]:
        indices[1:] = [-1,-1]
    elif maxima[2] < 0.1*maxima[0]:
        indices[2] = -1
    return tuple(indices)


def _record_orientation(histogram, first_angle, second_angle, index):
    rotation = np.float32(first_angle-second_angle)
    if rotation < 0:
        rotation += np.float32(360)
    # This is 1/30 in the actual source, not the usual 30/360 histogram factor.
    bin_index = int(math.floor(float(rotation*np.float32(1/30))+0.5))
    if bin_index == HISTOGRAM_LENGTH:
        bin_index = 0
    histogram[bin_index].append(index)


def _prune_orientation(matches, histogram, empty):
    retained = compute_three_maxima(histogram)
    for bin_index, indices in enumerate(histogram):
        if bin_index not in retained:
            for index in indices:
                matches[index] = empty


def _best_two(descriptor, descriptors, indices, initial_distance=256):
    """Stable strict '<' updates reproduce source best/second-best handling."""
    best_index, best, second = -1, initial_distance, initial_distance
    if len(indices):
        distances = hamming(descriptors[indices],descriptor)
        for index,distance in zip(indices,distances):
            if distance < best:
                second, best, best_index = best, int(distance), int(index)
            elif distance < second:
                second = int(distance)
    return best_index,best,second


def _global_area(frame, x, y, radius, min_level=-1, max_level=-1, right=False):
    indices = frame.get_features_in_area(x,y,radius,min_level,max_level,right)
    return indices+frame.n_left if right and frame.n_left >= 0 else indices


def _point_valid(point):
    return point is not None and not point.bad


def _geometry_candidate(frame, point, right=False, pose=None, view_limit=.5,
                        check_normal=True, positive_depth=True, pinhole_projection=False):
    side_index = frame.n_left if right else 0
    pose = frame.pose_for(side_index) if pose is None else pose
    camera = frame.camera_for(side_index)
    local = transform_points(pose,point.position)
    if (positive_depth and local[2] <= 0) or local[2] == 0:
        return None
    pixel = (local[:2]/local[2]*[camera.fx,camera.fy]+[camera.cx,camera.cy]
             if pinhole_projection else camera.project(local))
    if not np.isfinite(pixel).all() or not frame.in_image(*pixel):
        return None
    center = -pose[:3,:3].T @ pose[:3,3]
    direction = point.position-center
    distance = np.linalg.norm(direction)
    if distance < .8*point.min_distance or distance > 1.2*point.max_distance or distance == 0:
        return None
    view_cosine = direction @ point.normal/distance
    if check_normal and view_cosine < view_limit:
        return None
    level = point.predict_scale(distance,frame.scales)
    return pixel,local,distance,view_cosine,level


def search_for_initialization(first, second, previous_positions, window_size=100,
                              nnratio=.9, check_orientation=True):
    matches = np.full(len(first.features),-1,dtype=np.int32)
    reverse_matches = np.full(len(second.features),-1,dtype=np.int32)
    matched_distances = np.full(len(second.features),np.iinfo(np.int32).max,dtype=np.int32)
    histogram = [[] for _ in range(HISTOGRAM_LENGTH)]
    for i in range(len(first.features)):
        if first.features.octaves[i] > 0:
            continue
        indices = second.get_features_in_area(*previous_positions[i],window_size,0,0)
        descriptor = first.features.descriptors[i]
        if len(indices):
            distances = hamming(second.features.descriptors[indices],descriptor)
            indices = indices[matched_distances[indices] > distances]
        j,best,second_best = _best_two(descriptor,second.features.descriptors,indices,
                                     np.iinfo(np.int32).max)
        if j < 0 or best > TH_LOW or best >= nnratio*second_best:
            continue
        if reverse_matches[j] >= 0:
            matches[reverse_matches[j]] = -1
        matches[i], reverse_matches[j], matched_distances[j] = j,i,best
        if check_orientation:
            _record_orientation(histogram,first.features.angles[i],second.features.angles[j],i)
    if check_orientation:
        _prune_orientation(matches,histogram,-1)
    for i in np.flatnonzero(matches >= 0):
        previous_positions[i] = second.features.points[matches[i]]
    return matches


def search_by_bow_frame(keyframe, frame, nnratio=.7, check_orientation=True):
    matches = [None]*len(frame.features)
    histogram = [[] for _ in range(HISTOGRAM_LENGTH)]
    for node in sorted(keyframe.feature_nodes.keys() & frame.feature_nodes.keys()):
        for i in keyframe.feature_nodes[node]:
            point = keyframe.map_points[i]
            if not _point_valid(point):
                continue
            candidates = np.asarray([j for j in frame.feature_nodes[node] if matches[j] is None],dtype=int)
            left = candidates if frame.n_left < 0 else candidates[candidates < frame.n_left]
            j,best,second = _best_two(keyframe.features.descriptors[i],frame.features.descriptors,left)
            if best > TH_LOW:
                continue
            if best < nnratio*second:
                matches[j] = point
                if check_orientation:
                    _record_orientation(histogram,keyframe.features.angles[i],frame.features.angles[j],j)
            if frame.n_left >= 0:
                right = candidates[candidates >= frame.n_left]
                j,best,second = _best_two(keyframe.features.descriptors[i],frame.features.descriptors,right)
                # Source explicitly disables the right-camera ratio test with
                # "|| true" and nests this branch under the left TH_LOW gate.
                if best <= TH_LOW:
                    matches[j] = point
                    if check_orientation:
                        _record_orientation(histogram,keyframe.features.angles[i],frame.features.angles[j],j)
    if check_orientation:
        _prune_orientation(matches,histogram,None)
    return matches


def search_by_bow_keyframes(first, second, nnratio=.75, check_orientation=True):
    matches = [None]*len(first.features)
    used_second = np.zeros(len(second.features),dtype=bool)
    histogram = [[] for _ in range(HISTOGRAM_LENGTH)]
    for node in sorted(first.feature_nodes.keys() & second.feature_nodes.keys()):
        for i in first.feature_nodes[node]:
            # The source keyframe/keyframe overload restricts unrectified
            # rigs to the left-camera keys, unlike the frame overload.
            if first.n_left >= 0 and i >= first.n_left:
                continue
            if not _point_valid(first.map_points[i]):
                continue
            candidates = np.asarray([j for j in second.feature_nodes[node]
                                     if not used_second[j] and _point_valid(second.map_points[j])
                                     and (second.n_left < 0 or j < second.n_left)],dtype=int)
            j,best,second_best = _best_two(first.features.descriptors[i],second.features.descriptors,candidates)
            if best < TH_LOW and best < nnratio*second_best:
                matches[i], used_second[j] = second.map_points[j],True
                if check_orientation:
                    _record_orientation(histogram,first.features.angles[i],second.features.angles[j],i)
    if check_orientation:
        _prune_orientation(matches,histogram,None)
    return matches


def search_by_projection_local(frame, points, threshold=1., nnratio=.8,
                               far_points=False, far_threshold=np.inf):
    added = 0
    for point in points:
        if not _point_valid(point):
            continue
        left_projection = _geometry_candidate(frame,point)
        if left_projection is not None:
            point.track_depth = float(np.linalg.norm(left_projection[1]))
        if far_points and left_projection is not None and left_projection[2] > far_threshold:
            continue
        sides = [False,True] if frame.n_left >= 0 else [False]
        for right in sides:
            projection = _geometry_candidate(frame,point,right) if right else left_projection
            if projection is None:
                continue
            pixel,local,distance,view_cosine,level = projection
            radius = 2.5 if view_cosine > .998 else 4.
            # The source right-camera branch omits the external th factor.
            if not right:
                radius *= threshold
            radius *= frame.scales[level]
            candidates = _global_area(frame,*pixel,radius,level-1,level,right)
            selected = []
            for j in candidates:
                existing = frame.map_points[j]
                if existing is not None and existing.observation_count > 0:
                    continue
                if frame.n_left < 0 and frame.right_u[j] > 0:
                    predicted_right = pixel[0]-frame.baseline_fx/local[2]
                    if abs(predicted_right-frame.right_u[j]) > radius:
                        continue
                selected.append(j)
            candidates = np.asarray(selected,dtype=int)
            j,best,second = _best_two(point.descriptor,frame.features.descriptors,candidates)
            if j < 0 or best > TH_HIGH:
                continue
            # The ratio test applies only when best and second share octave.
            distances = hamming(frame.features.descriptors[candidates],point.descriptor)
            order = np.argsort(distances,kind="stable")
            second_level = int(frame.features.octaves[candidates[order[1]]]) if len(order)>1 and second < 256 else -1
            if frame.features.octaves[j] == second_level and best > nnratio*second:
                continue
            frame.map_points[j] = point
            added += 1
            if frame.n_left >= 0 and frame.stereo_matches is not None and frame.stereo_matches[j] >= 0:
                frame.map_points[frame.stereo_matches[j]] = point
                added += 1
    return added


def search_by_projection_last_frame(current, last, threshold=7., monocular=False,
                                    check_orientation=True):
    histogram = [[] for _ in range(HISTOGRAM_LENGTH)]
    current_center_in_last = transform_points(last.pose,current.center)
    forward = current_center_in_last[2] > current.baseline and not monocular
    backward = -current_center_in_last[2] > current.baseline and not monocular
    added = 0
    for i,point in enumerate(last.map_points):
        if point is None or last.outliers[i]:
            continue
        local = transform_points(current.pose,point.position)
        if local[2] <= 0:
            continue
        pixel = current.camera.project(local)
        if not current.in_image(*pixel):
            continue
        octave = int(last.features.octaves[i])
        radius = threshold*current.scales[octave]
        min_level,max_level = ((octave,-1) if forward else (0,octave) if backward
                               else (octave-1,octave+1))
        left_indices = _global_area(current,*pixel,radius,min_level,max_level)
        if not len(left_indices):
            continue
        for right in ([False,True] if current.n_left >= 0 else [False]):
            if right:
                local_right = transform_points(current.T_right_left,local)
                # The active source calls mpCamera (the LEFT model) here.
                right_pixel = current.camera.project(local_right)
                candidates = _global_area(current,*right_pixel,radius,min_level,max_level,True)
            else:
                candidates = left_indices
            selected = []
            for j in candidates:
                existing = current.map_points[j]
                if existing is not None and existing.observation_count > 0:
                    continue
                if current.n_left < 0 and current.right_u[j] > 0:
                    if abs(pixel[0]-current.baseline_fx/local[2]-current.right_u[j]) > radius:
                        continue
                selected.append(j)
            j,best,_ = _best_two(point.descriptor,current.features.descriptors,np.asarray(selected,dtype=int))
            if best <= TH_HIGH:
                current.map_points[j] = point
                added += 1
                if check_orientation:
                    _record_orientation(histogram,last.features.angles[i],current.features.angles[j],j)
    if check_orientation:
        before = sum(point is not None for point in current.map_points)
        _prune_orientation(current.map_points,histogram,None)
        added -= before-sum(point is not None for point in current.map_points)
    return added


def search_by_projection_relocalization(current, keyframe, already_found, threshold=3.,
                                        orb_distance=64, check_orientation=True):
    histogram = [[] for _ in range(HISTOGRAM_LENGTH)]
    already_ids = {id(point) for point in already_found if point is not None}
    added = 0
    for i,point in enumerate(keyframe.map_points):
        if not _point_valid(point) or id(point) in already_ids:
            continue
        projection = _geometry_candidate(current,point,check_normal=False,positive_depth=False)
        if projection is None:
            continue
        pixel,_,_,_,level = projection
        candidates = _global_area(current,*pixel,threshold*current.scales[level],level-1,level+1)
        candidates = np.asarray([j for j in candidates if current.map_points[j] is None],dtype=int)
        j,best,_ = _best_two(point.descriptor,current.features.descriptors,candidates)
        if best <= orb_distance:
            current.map_points[j] = point
            added += 1
            if check_orientation:
                _record_orientation(histogram,keyframe.features.angles[i],current.features.angles[j],j)
    if check_orientation:
        before = sum(point is not None for point in current.map_points)
        _prune_orientation(current.map_points,histogram,None)
        added -= before-sum(point is not None for point in current.map_points)
    return added


def search_for_triangulation(first, second, only_stereo=False, coarse=False,
                             check_orientation=False):
    matches = np.full(len(first.features),-1,dtype=np.int32)
    histogram = [[] for _ in range(HISTOGRAM_LENGTH)]
    center_in_second = transform_points(second.pose,first.center)
    epipole = second.camera.project(center_in_second) if center_in_second[2] != 0 else np.array([np.inf,np.inf])
    for node in sorted(first.feature_nodes.keys() & second.feature_nodes.keys()):
        for i in first.feature_nodes[node]:
            if first.map_points[i] is not None:
                continue
            stereo1 = first.right_camera is None and first.right_u[i] >= 0
            if only_stereo and not stereo1:
                continue
            best,j_best = TH_LOW,-1
            for j in second.feature_nodes[node]:
                if second.map_points[j] is not None:
                    continue
                stereo2 = second.right_camera is None and second.right_u[j] >= 0
                if only_stereo and not stereo2:
                    continue
                distance = int(hamming(first.features.descriptors[i],second.features.descriptors[j]))
                if distance > TH_LOW or distance > best:
                    continue
                if not stereo1 and not stereo2 and first.right_camera is None:
                    if np.sum((epipole-second.features.points[j])**2) < 100*second.scales[second.features.octaves[j]]:
                        continue
                relative = first.pose_for(i) @ inverse_pose(second.pose_for(j))
                camera1,camera2 = first.camera_for(i),second.camera_for(j)
                if coarse or camera1.epipolar_constrain(camera2,first.features.points[i],second.features.points[j],
                                                       relative[:3,:3],relative[:3,3],
                                                       1/first.inverse_variances[first.features.octaves[i]],
                                                       1/second.inverse_variances[second.features.octaves[j]]):
                    best,j_best = distance,j
            if j_best >= 0:
                # Source vbMatched2 is never set: many-to-one pairs are allowed.
                matches[i] = j_best
                if check_orientation:
                    _record_orientation(histogram,first.features.angles[i],second.features.angles[j_best],i)
    if check_orientation:
        _prune_orientation(matches,histogram,-1)
    return [(int(i),int(matches[i])) for i in np.flatnonzero(matches >= 0)]


def search_by_projection_sim3(keyframe, similarity, points, matches, threshold=3.,
                               hamming_ratio=1., point_keyframes=None, matched_keyframes=None):
    pose = similarity.camera_pose()
    already_found = {id(point) for point in matches if point is not None}
    added = 0
    for index,point in enumerate(points):
        if not _point_valid(point) or id(point) in already_found:
            continue
        projection = _geometry_candidate(keyframe,point,pose=pose,
                                         pinhole_projection=point_keyframes is not None)
        if projection is None:
            continue
        pixel,local,_,_,level = projection
        if point_keyframes is not None:
            # The provenance-carrying overload uses an explicit pinhole
            # projection rather than GeometricCamera::project in the source.
            pixel = local[:2]/local[2]*[keyframe.camera.fx,keyframe.camera.fy]+[keyframe.camera.cx,keyframe.camera.cy]
            if not keyframe.in_image(*pixel):
                continue
        candidates = _global_area(keyframe,*pixel,threshold*keyframe.scales[level],level-1,level)
        candidates = np.asarray([j for j in candidates if matches[j] is None],dtype=int)
        j,best,_ = _best_two(point.descriptor,keyframe.features.descriptors,candidates)
        if best <= TH_LOW*hamming_ratio:
            matches[j] = point
            if matched_keyframes is not None:
                matched_keyframes[j] = point_keyframes[index]
            added += 1
    return added


def search_by_sim3(first, second, matches12, similarity12, threshold=7.5):
    """Mutual projection agreement, S12 maps camera2 coordinates to camera1."""
    used1 = np.array([point is not None for point in matches12])
    used2 = np.zeros(len(second.features),dtype=bool)
    for point in matches12:
        if point is not None:
            index = point.observations.get(second.id,(-1,-1))[0]
            if 0 <= index < len(used2):
                used2[index] = True

    def direction(source,target,similarity,used):
        projected_matches = np.full(len(source.features),-1,dtype=int)
        for i,point in enumerate(source.map_points):
            if not _point_valid(point) or used[i]:
                continue
            local = similarity.map(transform_points(source.pose,point.position))
            if local[2] <= 0:
                continue
            # Both directions deliberately use KF1 intrinsics in this overload.
            pixel = local[:2]/local[2]*[first.camera.fx,first.camera.fy]+[first.camera.cx,first.camera.cy]
            if not target.in_image(*pixel):
                continue
            distance = np.linalg.norm(local)
            if distance < .8*point.min_distance or distance > 1.2*point.max_distance:
                continue
            level = point.predict_scale(distance,target.scales)
            candidates = _global_area(target,*pixel,threshold*target.scales[level],level-1,level)
            j,best,_ = _best_two(point.descriptor,target.features.descriptors,candidates,np.iinfo(np.int32).max)
            if best <= TH_HIGH:
                projected_matches[i] = j
        return projected_matches
    forward = direction(first,second,similarity12.inverse(),used1)
    backward = direction(second,first,similarity12,used2)
    added = 0
    for i,j in enumerate(forward):
        if j >= 0 and backward[j] == i:
            matches12[i] = second.map_points[j]
            added += 1
    return added


def fuse(keyframe, points, threshold=3., right=False):
    """Fuse projected observations; the point with more observations survives."""
    fused = 0
    for point in points:
        if not _point_valid(point) or keyframe.id in point.observations:
            continue
        projection = _geometry_candidate(keyframe,point,right)
        if projection is None:
            continue
        pixel,local,_,_,level = projection
        candidates = _global_area(keyframe,*pixel,threshold*keyframe.scales[level],level-1,level,right)
        eligible = []
        for j in candidates:
            residual = pixel-keyframe.features.points[j]
            error = residual @ residual
            cutoff = 5.99
            if keyframe.right_u[j] >= 0:
                error += (pixel[0]-keyframe.baseline_fx/local[2]-keyframe.right_u[j])**2
                cutoff = 7.8
            if error*keyframe.inverse_variances[keyframe.features.octaves[j]] <= cutoff:
                eligible.append(j)
        j,best,_ = _best_two(point.descriptor,keyframe.features.descriptors,np.asarray(eligible,dtype=int))
        if best > TH_LOW:
            continue
        existing = keyframe.map_points[j]
        if existing is not None:
            if not existing.bad:
                if existing.observation_count > point.observation_count:
                    point.replace(existing)
                else:
                    existing.replace(point)
        else:
            point.add_observation(keyframe,j)
            keyframe.map_points[j] = point
        fused += 1
    return fused


def fuse_sim3(keyframe, similarity, points, threshold=3.):
    """Return pending replacements; new unmatched observations are added now."""
    replacements = [None]*len(points)
    pose = similarity.camera_pose()
    already_found = {id(point) for point in keyframe.map_points if point is not None}
    fused = 0
    for i,point in enumerate(points):
        if not _point_valid(point) or id(point) in already_found:
            continue
        projection = _geometry_candidate(keyframe,point,pose=pose)
        if projection is None:
            continue
        pixel,_,_,_,level = projection
        candidates = _global_area(keyframe,*pixel,threshold*keyframe.scales[level],level-1,level)
        j,best,_ = _best_two(point.descriptor,keyframe.features.descriptors,candidates,np.iinfo(np.int32).max)
        if best > TH_LOW:
            continue
        existing = keyframe.map_points[j]
        if existing is not None:
            if not existing.bad:
                replacements[i] = existing
        else:
            point.add_observation(keyframe,j)
            keyframe.map_points[j] = point
        fused += 1
    return fused,replacements
