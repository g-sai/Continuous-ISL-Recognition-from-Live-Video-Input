def calculate_overlap_ratio(ground_truth_lists, combined_segment_lists):
    
    """
    Calculate Intersection over Union (IoU) for each segment in multiple combined segment lists.
    
    Args:
    ground_truth_lists (list of lists): List of ground truth segment lists
    combined_segment_lists (list of lists): List of combined segment lists
    
    Returns:
    list of lists: IoU values for each segment list
    float: Average IoU across all segment lists
    """

    all_iou_scores = []
    
    for ground_truth, combined_segments in zip(ground_truth_lists, combined_segment_lists):
        iou_scores = []
        
        for gt_seg, comb_seg in zip(ground_truth, combined_segments):
            gt_start, gt_end = gt_seg
            comb_start, comb_end = comb_seg
            
            intersection_start = max(gt_start, comb_start)
            intersection_end = min(gt_end, comb_end)
            intersection = max(0, intersection_end - intersection_start)
            
            union_start = min(gt_start, comb_start)
            union_end = max(gt_end, comb_end)
            union = union_end - union_start
            
            iou = intersection / union if union > 0 else 0
            iou_scores.append(iou)
        
        all_iou_scores.append(iou_scores)
    
    avg_iou_scores = [sum(iou_scores) / len(iou_scores) for iou_scores in all_iou_scores]
    avg_iou = sum(avg_iou_scores) / len(avg_iou_scores)
    
    return all_iou_scores, avg_iou


ground_truth_list = [
    [(), (), (), ()], # Each tuple has start_frame,end_frame
]

combined_segments_list = [
    [(), (), (), ()], # Each tuple has start_frame,end_frame
]


iou_scores, avg_iou = calculate_overlap_ratio(ground_truth_list, combined_segments_list)
print(f"   Individual IoU Scores: {iou_scores}")
print(f"   Average IoU: {avg_iou:.4f}")