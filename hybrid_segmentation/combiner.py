from hybrid_segmentation.velocity_segmentation import velocity_jerk_segmentation
from hybrid_segmentation.dwt_segmentation import dwt_segmentation
from hybrid_segmentation.basic_segmentation import predict_segments_with_padding,robust_segmentation



def combine_segmentations(video_features,y_min=0.7, y_max=1.0,consecutive_frames=5, min_gap=30, 
                         dwt_weight=0.6, velocity_weight=0.4, 
                         ):

    segments, _,boundaries = robust_segmentation(video_features, y_min, y_max, consecutive_frames, min_gap)
    normal_segments = predict_segments_with_padding(video_features, boundaries)
    num_segments=len(normal_segments)

    print("Normal segments : ",normal_segments)
    dwt_segments = dwt_segmentation(video_features, num_segments, min_segment_length=min_gap)
    print('DWT SEGMENTS : ',dwt_segments)
    velocity_segments = velocity_jerk_segmentation(video_features, num_segments, min_segment_length=min_gap)
    print('Velocity segments :',velocity_segments)
    
    total_frames = len(video_features)
    def segments_to_boundaries(segments):
        boundaries = set()
        for start, end in segments:
            if start > 0:
                boundaries.add(start)
            if end < total_frames:
                boundaries.add(end)
        return sorted(list(boundaries))
        
    dwt_boundaries = segments_to_boundaries(dwt_segments)
    velocity_boundaries = segments_to_boundaries(velocity_segments)
    def calculate_boundary_confidence(point, all_boundaries, window_size=30):
        confidence = 0
        for other_point in all_boundaries:
            if abs(point - other_point) <= window_size:
                proximity_score = 1 - (abs(point - other_point) / window_size)
                confidence = max(confidence, proximity_score)
        return confidence
    dwt_confidence = {point: calculate_boundary_confidence(point, velocity_boundaries) 
                     for point in dwt_boundaries}
    velocity_confidence = {point: calculate_boundary_confidence(point, dwt_boundaries) 
                         for point in velocity_boundaries} 
    all_potential_boundaries = set()
    boundary_scores = {}
    for point in dwt_boundaries:
        score = dwt_weight * (1 + dwt_confidence[point])  
        all_potential_boundaries.add(point)
        boundary_scores[point] = score
    for point in velocity_boundaries:
        score = velocity_weight * (1 + velocity_confidence[point])
        if point in boundary_scores:
            boundary_scores[point] += score  
        else:
            all_potential_boundaries.add(point)
            boundary_scores[point] = score
    
    sorted_boundaries = sorted(all_potential_boundaries, 
                             key=lambda x: boundary_scores[x], 
                             reverse=True)
    
    final_boundaries = [0]  
    for boundary in sorted_boundaries:
        valid_boundary = True
        for existing_boundary in final_boundaries:
            if abs(boundary - existing_boundary) < min_gap:
                valid_boundary = False
                break
        
        if valid_boundary:
            final_boundaries.append(boundary)
            if len(final_boundaries) == num_segments:
                break
    
    final_boundaries.append(total_frames) 
    final_boundaries.sort()
    
    final_segments = list(zip(final_boundaries[:-1], final_boundaries[1:]))
    

    return final_segments  
  