import numpy as np

def calculate_metrics_for_list(ground_truth_list, combined_segments_list):
   
    """
    Calculate metrics for multiple sample sets
    
    Parameters:
    ground_truth_list: List of ground truth segment lists
    combined_segments_list: List of combined segment lists
    
    Returns:
    Dictionary of metric results
    """

    results = {
        'iou': [],
        'boundary_displacement_error': [],
        'overlap_percentage': [],
        'f1_score': []
    }
    
    assert len(ground_truth_list) == len(combined_segments_list), "Lists must have equal length"
    
    for ground_truth, combined_segments in zip(ground_truth_list, combined_segments_list):
        def segment_to_array(segments, max_length):
            arr = np.zeros(max_length, dtype=bool)
            for start, end in segments:
                arr[start:end] = True
            return arr
        
        max_length = max(max(seg[1] for seg in ground_truth), 
                         max(seg[1] for seg in combined_segments))
        
        gt_arr = segment_to_array(ground_truth, max_length)
        combined_arr = segment_to_array(combined_segments, max_length)
        
        intersection = np.sum(gt_arr & combined_arr)
        union = np.sum(gt_arr | combined_arr)
        iou = intersection / union if union > 0 else 0
        results['iou'].append(iou)
        
        bde = np.mean([abs(gt[0] - comb[0]) + abs(gt[1] - comb[1]) 
                       for gt, comb in zip(ground_truth, combined_segments)]) / 2
        results['boundary_displacement_error'].append(bde)
        
        def segment_to_set(segments):
            point_set = set()
            for start, end in segments:
                point_set.update(range(start, end))
            return point_set
        
        gt_points = segment_to_set(ground_truth)
        combined_points = segment_to_set(combined_segments)
        
        overlapping_points = len(gt_points.intersection(combined_points))
        total_gt_points = len(gt_points)
        overlap_pct = (overlapping_points / total_gt_points) * 100 if total_gt_points > 0 else 0
        results['overlap_percentage'].append(overlap_pct)
        
        def find_closest_match(segment, segment_list):
            return min(segment_list, key=lambda x: abs(x[0] - segment[0]) + abs(x[1] - segment[1]))
        
        matched_segments = 0
        for gt_seg in ground_truth:
            closest = find_closest_match(gt_seg, combined_segments)
            if abs(gt_seg[0] - closest[0]) < 10 and abs(gt_seg[1] - closest[1]) < 10:
                matched_segments += 1
        
        precision = matched_segments / len(combined_segments) if combined_segments else 0
        recall = matched_segments / len(ground_truth) if ground_truth else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        results['f1_score'].append(f1)
    
    summary = {
        metric: {
            'mean': np.mean(values),
            'std': np.std(values)
        } for metric, values in results.items()
    }
    
    return results, summary

ground_truth_list = [
    [(), (), (), ()], # Each tuple has start_frame,end_frame
]

combined_segments_list = [
    [(), (), (), ()], # Each tuple has start_frame,end_frame
]


individual_results, summary_results = calculate_metrics_for_list(ground_truth_list, combined_segments_list)

print("Individual Results:")
for metric, values in individual_results.items():
    print(f"{metric}: {values}")

print("\nSummary Results:")
for metric, stats in summary_results.items():
    print(f"{metric}: Mean = {stats['mean']:.4f}, Std = {stats['std']:.4f}")